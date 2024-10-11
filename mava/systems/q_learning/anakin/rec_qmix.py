import copy
import functools
import time
from typing import Any, Callable, Dict, NamedTuple, Sequence, Tuple

import chex
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from flax.linen.initializers import orthogonal
from jax import Array
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from typing_extensions import TypeAlias

from jumanji.env import Environment, State
from mava.evaluator import ActorState, get_eval_fn, get_num_eval_envs
from mava.types import Observation, ObservationGlobalState, RNNObservation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import unreplicate_batch_dim
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import episode_metrics

from flax.linen import FrozenDict
from jumanji.types import TimeStep
from jax import tree
from mava.networks import ScannedRNN, RecQNetwork
from mava.utils.jax_utils import switch_leading_axes
from mava.utils.total_timestep_checker import check_total_timesteps

Metrics = Dict[str, Array]


class Transition(NamedTuple):
    """Transition for recurrent Q-learning."""

    obs: Observation
    action: Array
    reward: Array
    terminal: Array
    term_or_trunc: Array
    # Even though we use a trajectory buffer we need to store both obs and next_obs.
    # This is because of how the `AutoResetWrapper` returns obs at the end of an episode.
    next_obs: Observation


BufferState: TypeAlias = TrajectoryBufferState[Transition]


class QMIXParams(NamedTuple):
    online: FrozenVariableDict
    target: FrozenVariableDict
    mixer_online: FrozenVariableDict
    mixer_target: FrozenVariableDict


class LearnerState(NamedTuple):
    """State of the learner in an interaction-training loop."""

    # Interaction vars
    obs: Observation
    terminal: Array
    term_or_trunc: Array
    hidden_state: Array
    env_state: State
    time_steps: Array

    # Train vars
    train_steps: Array
    opt_state: optax.OptState

    # Shared vars
    buffer_state: TrajectoryBufferState
    params: QMIXParams
    key: PRNGKey


class ActionSelectionState(NamedTuple):
    online_params: FrozenVariableDict
    hidden_state: chex.Array
    time_steps: int
    key: chex.PRNGKey

class ActionState(NamedTuple):
    """The carry in the interaction loop."""

    action_selection_state: ActionSelectionState
    env_state: State
    buffer_state: BufferState
    obs: Observation
    terminal: Array
    term_or_trunc: Array


class TrainState(NamedTuple):  # 'carry' in training loop
    buffer_state: BufferState
    params: QMIXParams
    opt_state: optax.OptState
    train_steps: Array
    key: chex.PRNGKey


from flax.linen.initializers import lecun_normal


def _parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
    }
    return activation_fns[activation_fn_name]


def _parse_kernel_init_fn(
    kernel_init_fn_name: str,
) -> Callable[[chex.Array], chex.Array]:
    """Get kernel init function."""
    init_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "orthogonal": orthogonal(jnp.sqrt(2)),
        "lecun_normal": lecun_normal(),
    }
    return init_fns[kernel_init_fn_name]


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    kernel_init: str = "orthogonal"  # orthogonal or lecun_normal
    use_layer_norm: bool = False
    activate_final: bool = True

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)
        self.kernel_init_fn = _parse_kernel_init_fn(self.kernel_init)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(layer_size, kernel_init=self.kernel_init_fn)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)

            if i != len(self.layer_sizes) - 1:
                x = self.activation_fn(x)
            elif i == len(self.layer_sizes) - 1 and self.activate_final:
                x = self.activation_fn(x)

        return x


class QMixNetwork(nn.Module):
    num_actions: int
    num_agents: int
    hyper_hidden_dim: int = 64
    embed_dim: int = 32
    norm_env_states: bool = True

    def setup(self) -> None:
        self.hyper_w1: MLPTorso = MLPTorso(
            (self.hyper_hidden_dim, self.embed_dim * self.num_agents),
            activate_final=False,  # kernel_init="lecun_normal"
        )

        self.hyper_b1: MLPTorso = MLPTorso(
            (self.embed_dim,),  # kernel_init="lecun_normal"
        )

        self.hyper_w2: MLPTorso = MLPTorso(
            (self.hyper_hidden_dim, self.embed_dim),
            activate_final=False,  # kernel_init="lecun_normal"
        )

        self.hyper_b2: MLPTorso = MLPTorso(
            (self.embed_dim, 1),
            activate_final=False,  # kernel_init="lecun_normal"
        )

        self.layer_norm: nn.Module = nn.LayerNorm()

    @nn.compact
    def __call__(
        self,
        agent_qs: Array,
        env_global_state: Array,
    ) -> Array:
        B, T = agent_qs.shape[:2]  # batch size

        # # # Reshaping
        agent_qs = jnp.reshape(agent_qs, (B, T, 1, self.num_agents))

        if self.norm_env_states:
            states = self.layer_norm(env_global_state)
            # states = (env_global_state - jnp.min(env_global_state)) / (jnp.max(env_global_state) - jnp.min(env_global_state))
        else:
            states = env_global_state
        # states = jnp.ones_like(states) # NOTE debugging. remove this!!

        # First layer
        # w1 = cabs(self.hyper_w1(states)) # NOTE alternative absolute value calculation that has grad 0 at 0
        w1 = jnp.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = jnp.reshape(w1, (B, T, self.num_agents, self.embed_dim))
        b1 = jnp.reshape(b1, (B, T, 1, self.embed_dim))

        # Matrix multiplication
        hidden = nn.elu(jnp.matmul(agent_qs, w1) + b1)

        # Second layer
        # w2 = cabs(self.hyper_w2(states))
        w2 = jnp.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = jnp.reshape(w2, (B, T, self.embed_dim, 1))
        b2 = jnp.reshape(b2, (B, T, 1, 1))

        # Compute final output
        y = jnp.matmul(hidden, w2) + b2

        # Reshape
        q_tot = jnp.reshape(y, (B, T, 1))

        return q_tot


def init(
    cfg: DictConfig,
) -> Tuple[
    Tuple[Environment, Environment],  # jax.debug.print("{x}", x=q_loss_info["q_loss"])
    LearnerState,
    RecQNetwork,
    optax.GradientTransformation,
    TrajectoryBuffer,
    MavaLogger,
    chex.PRNGKey,
]:
    
    logger = MavaLogger(cfg)

    # init key, get devices available
    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = jax.tree_map(
            lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x
        )
        return jax.device_put_replicated(x, devices)

    # make envs
    env, eval_env = environments.make(cfg, add_global_state=True)

    action_dim = env.action_dim
    num_agents = env.num_agents

    key, q_key = jax.random.split(key, 2)

    # Shape legend:
    # T: Time (dummy dimension size = 1)
    # B: Batch (dummy dimension size = 1)
    # A: Agent
    # Make dummy inputs to init recurrent Q network -> need shape (T, B, A, ...)
    init_obs = env.observation_spec().generate_value()  # (A, ...)
    # (B, T, A, ...)
    init_obs_batched = tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], init_obs)
    init_term_or_trunc = jnp.zeros((1, 1, 1), dtype=bool)  # (T, B, 1)
    init_x = (init_obs_batched, init_term_or_trunc)  # pack the RNN dummy inputs
    # (B, A, ...)
    init_hidden_state = ScannedRNN.initialize_carry(
        (cfg.arch.num_envs, num_agents), cfg.network.hidden_state_dim
    )

    # Making recurrent Q network
    q_net = RecQNetwork(
        pre_torso=MLPTorso((256,)),
        post_torso=MLPTorso((256,)),
        num_actions=action_dim,
        hidden_state_dim=cfg.network.hidden_state_dim,)
    q_params = q_net.init(q_key, init_hidden_state, init_x)
    q_target_params = q_net.init(q_key, init_hidden_state, init_x)

    # Make QMixer
    dummy_agent_qs = jnp.zeros(
        (
            cfg.system.sample_batch_size,
            cfg.system.sample_sequence_length - 1,
            num_agents,
        ),
        "float32",
    )
    global_env_state_shape = (
        env.observation_spec().generate_value().global_state[0, :].shape
    )  # NOTE env wrapper currently duplicates env state for each agent
    dummy_global_env_state = jnp.zeros(
        (
            cfg.system.sample_batch_size,
            cfg.system.sample_sequence_length - 1,
            *global_env_state_shape,
        ),
        "float32",
    )
    q_mixer = QMixNetwork(action_dim, num_agents, 64, cfg.system.qmix_embed_dim)
    mixer_online_params = q_mixer.init(q_key, dummy_agent_qs, dummy_global_env_state)
    mixer_target_params = q_mixer.init(q_key, dummy_agent_qs, dummy_global_env_state)

    # Pack params
    params = QMIXParams(
        q_params, q_target_params, mixer_online_params, mixer_target_params
    )

    # OPTIMISER
    opt = optax.chain(
        optax.adam(learning_rate=cfg.system.q_lr),
    )
    opt_state = opt.init((params.online, params.mixer_online))

    # Distribute params and opt states across all devices
    params = replicate(params)
    opt_state = replicate(opt_state)
    init_hidden_state = replicate(init_hidden_state)

    # BUFFER CREATION INITS
    init_acts = env.action_spec().generate_value()  # (A,)
    init_transition = Transition(
        obs=init_obs,  # (A, ...)
        action=init_acts,
        reward=jnp.zeros((num_agents,), dtype=float),
        terminal=jnp.zeros((1,), dtype=bool),  # one flag for all agents
        term_or_trunc=jnp.zeros((1,), dtype=bool),
        next_obs=init_obs,
    )

    # Initialise trajectory buffer
    rb = fbx.make_trajectory_buffer(
        # n transitions gives n-1 full data points
        sample_sequence_length=cfg.system.sample_sequence_length + 1,
        period=1,  # sample any unique trajectory
        add_batch_size=cfg.arch.num_envs,
        sample_batch_size=cfg.system.sample_batch_size,
        max_length_time_axis=cfg.system.buffer_size,
        min_length_time_axis=cfg.system.min_buffer_size,
    )
    buffer_state = rb.init(init_transition)
    buffer_state = replicate(buffer_state)

    # Reset env
    n_keys = cfg.arch.num_envs * cfg.arch.n_devices * cfg.system.update_batch_size
    key_shape = (cfg.arch.n_devices, cfg.system.update_batch_size, cfg.arch.num_envs, -1)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_keys)
    reset_keys = jnp.reshape(reset_keys, key_shape)

    # Get initial state and timestep per-device
    env_state, first_timestep = jax.pmap(  # devices
        jax.vmap(  # update_batch_size
            jax.vmap(env.reset),  # num_envs
            axis_name="batch",
        ),
        axis_name="device",
    )(reset_keys)
    first_obs = first_timestep.observation
    first_term_or_trunc = first_timestep.last()[..., jnp.newaxis]
    first_term = (1 - first_timestep.discount[..., 0, jnp.newaxis]).astype(bool)
    
    # Initialise env steps and training steps
    t0_act = jnp.zeros((cfg.arch.n_devices, cfg.system.update_batch_size), dtype=int)
    t0_train = jnp.zeros((cfg.arch.n_devices, cfg.system.update_batch_size), dtype=int)

    # Keys passed to learner
    first_keys = jax.random.split(key, (cfg.arch.n_devices * cfg.system.update_batch_size))
    first_keys = first_keys.reshape((cfg.arch.n_devices, cfg.system.update_batch_size, -1))

    # Initial learner state.
    learner_state = LearnerState(
        first_obs,
        first_term,
        first_term_or_trunc,
        init_hidden_state,
        env_state,
        t0_act,
        t0_train,
        opt_state,
        buffer_state,
        params,
        first_keys,
    )

    return (env, eval_env), learner_state, q_net, q_mixer, opt, rb, logger, key


def make_update_fns(
    cfg: DictConfig,
    env: Environment,
    q_net: RecQNetwork,
    mixer: QMixNetwork,
    opt: optax.GradientTransformation,
    rb: TrajectoryBuffer,
) -> Any:

    def select_random_action_batch(
        selection_key: chex.PRNGKey, obs: ObservationGlobalState
    ) -> Tuple[chex.Array, chex.Array]:
        """Select actions randomly with masking."""

        def select_single_random_action(key, action_options, p_values):
            """Select one action per row with probabilities used for masking."""
            action = jax.random.choice(key, action_options, p=p_values)
            return action

        action_mask = obs.action_mask

        batch_size, n_agents, n_actions = action_mask.shape

        # get one key per agent
        keys = jax.random.split(selection_key, batch_size * n_agents)
        # base of action indexes
        action_options = jnp.arange(n_actions)
        # get num avail actions to generate probabilities for choosing
        num_avail_per_agent = jnp.sum(action_mask, axis=-1)[..., jnp.newaxis]
        # get probabilities (uniform with "masking")
        p_vals = action_mask.astype("int32") / num_avail_per_agent
        # flatten so that we only need one vmap over batch, agent dim
        p_vals = jnp.reshape(p_vals, (batch_size * n_agents, n_actions))
        # vmap random selection
        actions = jax.vmap(select_single_random_action, (0, None, 0))(
            keys, action_options, p_vals
        )
        # unflatten to mtch expected action block size
        actions = jnp.reshape(actions, (batch_size, n_agents))

        return actions

    # INTERACT LEVEL 2 (1.2)
    def select_eps_greedy_action(
        action_selection_state: ActionSelectionState,
        obs: ObservationGlobalState,
        term_or_trunc: Array,
    ) -> Tuple[ActionSelectionState, Array]:
        """Select action to take in eps-greedy way. Batch and agent dims are included."""

        params, hidden_state, t, key = action_selection_state
        
        eps = jnp.maximum(
            cfg.system.eps_min, 1 - (t / cfg.system.eps_decay) * (1 - cfg.system.eps_min)
        )

        obs = tree.map(lambda x: x[jnp.newaxis, ...], obs)
        term_or_trunc = tree.map(lambda x: x[jnp.newaxis, ...], term_or_trunc)

        next_hidden_state, eps_greedy_dist = q_net.apply(
            params, hidden_state, (obs, term_or_trunc), eps
        )

        # Random key splitting
        new_key, explore_key = jax.random.split(key, 2)

        action = eps_greedy_dist.sample(seed=explore_key)
        action = action[0, ...]  # (1, B, A) -> (B, A)

        # repack new selection params
        next_action_selection_state = ActionSelectionState(
            params, next_hidden_state, t + cfg.arch.num_envs, new_key
        )
        return next_action_selection_state, action

    def action_step(action_state: ActionState, _: Any) -> Tuple[ActionState, Dict]:
        """Selects an action, steps global env, stores timesteps in global rb and repacks the parameters for the next step."""

        # Unpack
        action_selection_state, env_state, buffer_state, obs, terminal, term_or_trunc = action_state

        # select the actions to take
        next_action_selection_state, action = select_eps_greedy_action(
            action_selection_state, obs, term_or_trunc
        )

        # step env with selected actions
        next_env_state, next_timestep = jax.vmap(env.step)(env_state, action)

        # Get reward
        reward = jnp.mean(
            next_timestep.reward, axis=-1, keepdims=True
        )  # NOTE (ruan): combine agent rewards, different to IQL.

        transition = Transition(
            obs, action, reward, terminal, term_or_trunc, next_timestep.extras["real_next_obs"]
        )
        # Add dummy time dim
        transition = tree.map(lambda x: x[:, jnp.newaxis, ...], transition)
        next_buffer_state = rb.add(buffer_state, transition)

        # Nexts
        next_obs = next_timestep.observation
        # make compatible with network input and transition storage in next step
        next_terminal = (1 - next_timestep.discount[..., 0, jnp.newaxis]).astype(bool)
        next_term_or_trunc = next_timestep.last()[..., jnp.newaxis]

        # Repack
        new_act_state = ActionState(
            next_action_selection_state,
            next_env_state,
            next_buffer_state,
            next_obs,
            next_terminal,
            next_term_or_trunc,
        )

        return new_act_state, next_timestep.extras["episode_metrics"]


    def prep_inputs_to_scannedrnn(obs: Observation, term_or_trunc: chex.Array) -> chex.Array:
        """Prepares the inputs to the RNN network for either getting q values or the
        eps-greedy distribution.

        Mostly swaps leading axes because the replay buffer outputs (B, T, ... )
        and the RNN takes in (T, B, ...).
        """
        hidden_state = ScannedRNN.initialize_carry(
            (cfg.system.sample_batch_size, obs.agents_view.shape[2]), cfg.network.hidden_state_dim
        )
        # the rb outputs (B, T, ... ) the RNN takes in (T, B, ...)
        obs = switch_leading_axes(obs)  # (B, T) -> (T, B)
        term_or_trunc = switch_leading_axes(term_or_trunc)  # (B, T) -> (T, B)
        obs_term_or_trunc = (obs, term_or_trunc)

        return hidden_state, obs_term_or_trunc

    # TRAIN LEVEL 3 (2.1.1)
    def q_loss_fn(
        online_params: FrozenVariableDict,
        obs: Array,
        term_or_trunc: Array,
        action: Array,
        target: Array,
    ) -> Tuple[Array, Metrics]:
        """The portion of the calculation to grad, namely online apply and mse with target."""
        q_online_params, online_mixer_params = online_params

        # axes switched here to scan over time
        hidden_state, obs_term_or_trunc = prep_inputs_to_scannedrnn(obs, term_or_trunc)

        # get online q values of all actions
        _, q_online = q_net.apply(
            q_online_params, hidden_state, obs_term_or_trunc, method="get_q_values"
        )
        q_online = switch_leading_axes(q_online)  # (T, B, ...) -> (B, T, ...)
        # get the q values of the taken actions and remove extra dim
        q_online = jnp.squeeze(
            jnp.take_along_axis(q_online, action[..., jnp.newaxis], axis=-1), axis=-1
        )

        q_online = mixer.apply(
            online_mixer_params, q_online, obs.global_state[:, :, 0, ...]
        )  # B,T,A,... -> B,T,1,... # NOTE states are replicated over agents thats why we only take first one

        q_loss = jnp.mean((q_online - target) ** 2)

        loss_info = {
            "q_loss": q_loss,
            "mean_q": jnp.mean(q_online),
            "max_q_error": jnp.max(jnp.abs(q_online - target) ** 2),
            "min_q_error": jnp.min(jnp.abs(q_online - target) ** 2),
            "mean_target": jnp.mean(target),
        }

        return q_loss, loss_info

    def update_q(
        params: QMIXParams, opt_states: optax.OptState, data: Transition, t_train: int
    ) -> Tuple[QMIXParams, optax.OptState, Metrics]:
        """Update the Q parameters."""

        # Get data aligned with current/next timestep
        data_first: Dict[str, chex.Array] = jax.tree_map(lambda x: x[:, :-1, ...], data)
        data_next: Dict[str, chex.Array] = jax.tree_map(lambda x: x[:, 1:, ...], data)

        obs = data_first.obs
        term_or_trunc = data_first.term_or_trunc
        reward = data_first.reward
        action = data_first.action

        # The three following variables all come from the same time step.
        # They are stored and accessed in this way because of the `AutoResetWrapper`.
        # At the end of an episode `data_first.next_obs` and `data_next.obs` will be
        # different, which is why we need to store both. Thus `data_first.next_obs`
        # aligns with the `terminal` from `data_next`.
        next_obs = data_first.next_obs
        next_term_or_trunc = data_next.term_or_trunc
        next_terminal = data_next.terminal

        # Scan over each sample
        hidden_state, next_obs_term_or_trunc = prep_inputs_to_scannedrnn(
            next_obs, next_term_or_trunc
        )

        ###############
        # OLD CODE
        ###############
        # # Scan over each sample and discard first timestep
        # next_q_vals_online = scan_apply(params.online, data.obs, data.done)
        # next_q_vals_online = jnp.where(
        #     data.obs.action_mask,
        #     next_q_vals_online,
        #     jnp.zeros(next_q_vals_online.shape) - 9999999,
        # )[:, 1:, ...]
        # next_q_vals_target = scan_apply(params.target, data.obs, data.done)[:, 1:, ...]

        # eps defaults to 0
        _, next_online_greedy_dist = q_net.apply(
            params.online, hidden_state, next_obs_term_or_trunc
        )

        _, next_q_vals_target = q_net.apply(
            params.target, hidden_state, next_obs_term_or_trunc, method="get_q_values"
        )

        # Get the greedy action
        next_action = next_online_greedy_dist.mode()  # (T, B, ...)

        # Double q-value selection
        next_q_val = jnp.squeeze(
            jnp.take_along_axis(next_q_vals_target, next_action[..., jnp.newaxis], axis=-1), axis=-1
        )

        next_q_val = switch_leading_axes(next_q_val)  # (T, B, ...) -> (B, T, ...)

        next_q_val = mixer.apply(
            params.mixer_target, next_q_val, data_next.obs.global_state[:, :, 0, ...]
        )  # B,T,A,... -> B,T,1,...

        # TD Target
        target_q_val = reward + (1.0 - next_terminal) * cfg.system.gamma * next_q_val

        # Update Q function.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            (params.online, params.mixer_online),
            obs,
            term_or_trunc,
            action,
            target_q_val,
        )
        q_loss_info["mean_first_reward"] = jnp.mean(reward)
        q_loss_info["mean_next_qval"] = jnp.mean(next_q_val)
        q_loss_info["done"] = jnp.mean(data.term_or_trunc)

        # Mean over the device and batch dimension.
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_updates, next_opt_state = opt.update(q_grads, opt_states)
        (next_online_params, next_mixer_params) = optax.apply_updates(
            (params.online, params.mixer_online), q_updates
        )

        # # Target network polyak update.
        # next_target_params = jax.lax.select(
        #     cfg.system.hard_update,
        next_target_params = optax.periodic_update(
            next_online_params, params.target, t_train, cfg.system.update_period
        )
        next_mixer_target_params = optax.periodic_update(
            next_mixer_params, params.mixer_target, t_train, cfg.system.update_period
        )
        #     optax.incremental_update(next_online_params, params.target, cfg.system.tau)
        # )
        # chex.assert_trees_all_equal(params.mixer_online, next_mixer_params)
        # jax.debug.print("{x}", x=q_loss_info["q_loss"])
        # Repack params and opt_states.
        next_params = QMIXParams(
            next_online_params,
            next_target_params,
            next_mixer_params,
            next_mixer_target_params,
        )

        return next_params, next_opt_state, q_loss_info

    # TRAIN LEVEL 1 (2.)
    def train(train_state: TrainState, _: Any) -> TrainState:
        """Sample, train and repack."""

        # unpack and get keys
        buffer_state, params, opt_states, t_train, key = train_state
        next_key, buff_key = jax.random.split(key, 2)

        # sample
        data = rb.sample(buffer_state, buff_key).experience

        # learn
        next_params, next_opt_states, q_loss_info = update_q(
            params, opt_states, data, t_train
        )

        next_train_state = TrainState(
            buffer_state, next_params, next_opt_states, t_train + 1, next_key
        )

        return next_train_state, q_loss_info

    # ___________________________________________________________________________________________________
    # INTERACT-TRAIN LOOP

    scanned_act = lambda state: lax.scan(action_step, state, None, length=cfg.system.rollout_length)
    scanned_train = lambda state: lax.scan(train, state, None, length=cfg.system.epochs)

    # interact and train
    def update_step(
        learner_state: LearnerState, _: Any
    ) -> Tuple[LearnerState, Tuple[Metrics, Metrics]]:
        """Interact, then learn. The _ at the end of a var means updated."""

        # unpack and get random keys
        (
            obs,
            terminal,
            term_or_trunc,
            hidden_state,
            env_state,
            time_steps,
            train_steps,
            opt_state,
            buffer_state,
            params,
            key,
        ) = learner_state
        new_key, interact_key, train_key = jax.random.split(key, 3)

        # Select actions, step env and store transitions
        action_selection_state = ActionSelectionState(
            params.online, hidden_state, time_steps, interact_key
        )
        action_state = ActionState(
            action_selection_state, env_state, buffer_state, obs, terminal, term_or_trunc
        )
        final_action_state, metrics = scanned_act(action_state)

        # Sample and learn
        train_state = TrainState(
            final_action_state.buffer_state,
            params,
            opt_state,
            train_steps,
            train_key,
        )
        final_train_state, losses = scanned_train(train_state)

        next_learner_state = LearnerState(
            final_action_state.obs,
            final_action_state.terminal,
            final_action_state.term_or_trunc,
            final_action_state.action_selection_state.hidden_state,
            final_action_state.env_state,
            final_action_state.action_selection_state.time_steps,
            final_train_state.train_steps,
            final_train_state.opt_state,
            final_action_state.buffer_state,
            final_train_state.params,
            new_key,
        )

        return next_learner_state, (metrics, losses)


    pmaped_updated_step = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(update_step, state, None, length=cfg.system.scan_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )

    return pmaped_updated_step


def run_experiment(cfg: DictConfig) -> float:
    cfg.arch.n_devices = len(jax.devices())
    cfg = check_total_timesteps(cfg)

    # Number of env steps before evaluating/logging.
    steps_per_rollout = int(cfg.system.total_timesteps // cfg.arch.num_evaluation)
    # Multiplier for a single env/learn step in an anakin system
    anakin_steps = cfg.arch.n_devices * cfg.system.update_batch_size
    # Number of env steps in one anakin style update.
    anakin_act_steps = anakin_steps * cfg.arch.num_envs * cfg.system.rollout_length
    # Number of steps to do in the scanned update method (how many anakin steps).
    cfg.system.scan_steps = int(steps_per_rollout / anakin_act_steps)

    pprint(OmegaConf.to_container(cfg, resolve=True))

    # Initialise system and make learning/evaluation functions
    (env, eval_env), learner_state, q_net, mixer, opts, rb, logger, key = init(cfg)
    update = make_update_fns(cfg, env, q_net, mixer, opts, rb)

    cfg.system.num_agents = env.num_agents

    key, eval_key = jax.random.split(key)

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: chex.PRNGKey, actor_state: ActorState
    ) -> Tuple[chex.Array, ActorState]:
        """The acting function that get's passed to the evaluator.
        A custom function is needed for epsilon-greedy acting.
        """
        hidden_state = actor_state["hidden_state"]

        term_or_trunc = timestep.last()
        net_input = (timestep.observation, term_or_trunc[..., jnp.newaxis])
        net_input = tree.map(lambda x: x[jnp.newaxis], net_input)  # add batch dim to obs
        next_hidden_state, eps_greedy_dist = q_net.apply(params, hidden_state, net_input)
        action = eps_greedy_dist.sample(seed=key).squeeze(0)
        return action, {"hidden_state": next_hidden_state}
    
    evaluator = get_eval_fn(eval_env, eval_act_fn, cfg, absolute_metric=False)

    if cfg.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=cfg,  # Save all config as metadata in the checkpoint
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(cfg, absolute_metric=False)
    eval_hs = ScannedRNN.initialize_carry(
        (jax.device_count(), eval_batch_size, cfg.system.num_agents),
        cfg.network.hidden_state_dim,
    )
    
    max_episode_return = -jnp.inf
    best_params = copy.deepcopy(unreplicate_batch_dim(learner_state.params.online))

    # Main loop:
    for eval_idx, t in enumerate(
        range(steps_per_rollout, int(cfg.system.total_timesteps + 1), steps_per_rollout)
    ):
        # Learn loop:
        start_time = time.time()
        learner_state, (metrics, losses) = update(learner_state)
        jax.block_until_ready(learner_state)

        # Log:
        # Multiply by learn steps here because anakin steps per second is learn + act steps
        # But we want to make sure we're counting env steps correctly so it's not included
        # in the loop counter.
        elapsed_time = time.time() - start_time
        eps = jnp.maximum(
            cfg.system.eps_min, 1 - (t / cfg.system.eps_decay) * (1 - cfg.system.eps_min)
        )
        final_metrics, ep_completed = episode_metrics.get_final_step_metrics(metrics)
        final_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
        loss_metrics = losses
        logger.log(
            {"timestep": t, "epsilon": eps},
            t,
            eval_idx,
            LogEvent.MISC,
        )
        if ep_completed:
            logger.log(final_metrics, t, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t, eval_idx, LogEvent.TRAIN)

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, cfg.arch.n_devices)
        eval_params = unreplicate_batch_dim(learner_state.params.online)
        eval_metrics = evaluator(eval_params, eval_keys, {"hidden_state": eval_hs})
        jax.block_until_ready(eval_metrics)
        logger.log(eval_metrics, t, eval_idx, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        # # Log:
        # episode_return = jnp.mean(eval_output.episode_metrics["episode_return"])
        # logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.EVAL)

        # Save best actor params.
        if cfg.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(eval_params)
            max_episode_return = episode_return

        # # Checkpoint:
        # if cfg.logger.checkpointing.save_model:
        #     # Save checkpoint of learner state
        #     unreplicated_learner_state = unreplicate_learner_state(learner_state)  # type: ignore
        #     checkpointer.save(
        #         timestep=t,
        #         unreplicated_learner_state=unreplicated_learner_state,
        #         episode_return=episode_return,
        #     )

    # Measure absolute metric.
    if cfg.arch.absolute_metric:
        eval_keys = jax.random.split(key, cfg.arch.n_devices)
        eval_batch_size = get_num_eval_envs(cfg, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (jax.device_count(), eval_batch_size, cfg.system.num_agents),
            cfg.network.hidden_state_dim,
        )

        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, cfg, absolute_metric=True)
        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": eval_hs})
        logger.log(eval_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    return float(episode_return)


@hydra.main(
    config_path="../../../configs/default/",
    config_name="rec_qmix.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "rec_qmix"
    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}QMIX experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()