# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools
import time
from typing import Any, Callable, Dict, NamedTuple, Sequence, Tuple

import chex
import distrax
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
from jax import Array, custom_jvp
from jax.typing import ArrayLike
from jumanji.env import Environment, State
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypeAlias

from mava.evaluator import make_eval_fns
from mava.types import Observation, ObservationGlobalState, RNNObservation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import unreplicate_batch_dim
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import episode_metrics

Metrics = Dict[str, Array]


class Transition(NamedTuple):
    obs: Array
    action: Array
    reward: Array
    done: Array


BufferState: TypeAlias = TrajectoryBufferState[Transition]


class QMIXParams(NamedTuple):
    online: FrozenVariableDict
    target: FrozenVariableDict
    mixer_online: FrozenVariableDict
    mixer_target: FrozenVariableDict


class LearnerState(NamedTuple):
    # Interaction vars
    obs: Array
    done: Array
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


class InteractionState(NamedTuple):  # 'carry' in interaction loop
    action_selection_state: ActionSelectionState
    env_state: State
    buffer_state: BufferState
    obs: Observation
    done: Array


class TrainState(NamedTuple):  # 'carry' in training loop
    buffer_state: BufferState
    params: QMIXParams
    opt_state: optax.OptState
    train_steps: Array
    key: chex.PRNGKey


## NOTE: Alternative absolute value calculation that has grad 0 at 0, jnp.abs has grad 1 at 0, tf.abs has grad 0 at 0
# @custom_jvp
# def cabs(x):
#   return jnp.abs(x)

# @cabs.defjvp
# def cabs_jvp(primals, tangents):
#   x, = primals
#   x_dot, = tangents
#   ans = jnp.abs(x)
#   ans_dot = jax.lax.select(
#       x==0.0,
#       jnp.zeros_like(x),
#       jnp.sign(x)
#   ) * x_dot
#   return ans, ans_dot

from flax.linen.initializers import lecun_normal, orthogonal


def _parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
    }
    return activation_fns[activation_fn_name]


def _parse_kernel_init_fn(kernel_init_fn_name: str) -> Callable[[chex.Array], chex.Array]:
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


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, :, jnp.newaxis],  # NOTE (Louise) changes about to hit in mava
            self.initialize_carry((ins.shape[0], ins.shape[1]), ins.shape[2]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: Sequence[int], hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )  # NOTE (Louise) this change has not yet been merged into develop, but is necessary to remove agent vmapping


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


class RecQNetwork(nn.Module):
    num_actions: int
    pre_torso: MLPTorso = MLPTorso(
        (256,)
    )  # NOTE (Louise) check that this is okay for always (uses default nw params)
    post_torso: MLPTorso = MLPTorso(
        (256,)
    )  # NOTE (Louise) check that this is okay for always (uses default nw params)

    @nn.compact
    def __call__(
        self,
        hidden_state: chex.Array,
        observations_resets: RNNObservation,
    ) -> Array:
        # unpack consumed parameters
        obs, resets = observations_resets

        embedding = self.pre_torso(obs.agents_view)

        # pack consumed parameters
        rnn_input = (embedding, resets)
        hidden_state, embedding = ScannedRNN()(hidden_state, rnn_input)

        embedding = self.post_torso(embedding)

        q_values = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(embedding)

        return hidden_state, q_values


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

    hidden_size = 256  # TODO: dynamically get?
    logger = MavaLogger(cfg)

    # init key, get devices available
    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()
    n_devices = len(devices)

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = jax.tree_map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    # make envs
    env, eval_env = environments.make(cfg, add_global_state=True)

    # actions, agents, keysplits
    num_actions = int(env.action_spec().num_values[0])
    num_agents = env.action_spec().shape[0]
    cfg.system.num_agents = num_agents
    key, q_key = jax.random.split(key, 2)

    # ALL INITS TO MAKE NETWORK -> need TBAx
    # initialise observation for the sizes
    init_obs = env.observation_spec().generate_value()  # A,x
    init_obs_batched = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], cfg.arch.num_envs, axis=0),
        init_obs,
    )  # n_envs, A , x
    init_obs_batched = jax.tree_util.tree_map(
        lambda x: x[jnp.newaxis, ...], init_obs_batched
    )  # add time dim (1,n_envs,A,x)
    init_done = jnp.zeros((1, cfg.arch.num_envs, 1), dtype=bool)  # (1,n_envs, 1)

    init_x = (init_obs_batched, init_done)
    # Initialise hidden states.
    init_hidden_state = ScannedRNN.initialize_carry(
        (cfg.arch.num_envs, num_agents), hidden_size
    )  # (1, n_envs, A, x)

    # Making recurrent Q network
    q_net = RecQNetwork(num_actions)
    q_params = q_net.init(q_key, init_hidden_state, init_x)
    q_target_params = q_net.init(q_key, init_hidden_state, init_x)

    # Make QMixer
    dummy_agent_qs = jnp.zeros(
        (cfg.system.sample_batch_size, cfg.system.sample_sequence_length - 1, num_agents), "float32"
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
    q_mixer = QMixNetwork(num_actions, num_agents, 64, cfg.system.qmix_embed_dim)
    mixer_online_params = q_mixer.init(q_key, dummy_agent_qs, dummy_global_env_state)
    mixer_target_params = q_mixer.init(q_key, dummy_agent_qs, dummy_global_env_state)

    # Pack params
    params = QMIXParams(q_params, q_target_params, mixer_online_params, mixer_target_params)

    # OPTIMISER
    opt = optax.chain(
        optax.adam(learning_rate=cfg.system.q_lr),  # eps=1e-5 in JAXMARL paper
    )
    opt_state = opt.init((params.online, params.mixer_online))

    # Distribute params and opt states across all devices
    params = replicate(params)
    opt_state = replicate(opt_state)
    init_hidden_state = replicate(init_hidden_state)

    # BUFFER CREATION INITS
    init_acts = env.action_spec().generate_value()  # A,
    init_obs = env.observation_spec().generate_value()  # A, x

    # Create replay buffer
    init_transition = Transition(
        obs=ObservationGlobalState(*init_obs),
        action=init_acts,
        reward=jnp.zeros((1,), dtype=float),
        done=jnp.zeros((1,), dtype=bool),
    )

    # Training rollout length NOTE (Claude) maybe rename this.
    rec_chunk_size = cfg.system.sample_sequence_length

    rb = fbx.make_trajectory_buffer(
        sample_sequence_length=rec_chunk_size,
        period=1,
        add_batch_size=cfg.arch.num_envs,  # TODO: is this accurate?
        sample_batch_size=cfg.system.sample_batch_size,
        max_length_time_axis=cfg.system.buffer_size,
        min_length_time_axis=cfg.system.min_buffer_size,
    )
    buffer_state = rb.init(init_transition)
    buffer_state = replicate(buffer_state)

    # Reset env
    n_keys = cfg.arch.num_envs * n_devices * cfg.system.update_batch_size
    key_shape = (n_devices, cfg.system.update_batch_size, cfg.arch.num_envs, -1)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_keys)
    reset_keys = jnp.reshape(reset_keys, key_shape)

    # Keys passed to learner
    first_keys = jax.random.split(key, (n_devices * cfg.system.update_batch_size))
    first_keys = first_keys.reshape((n_devices, cfg.system.update_batch_size, -1))

    env_state, first_timestep = jax.pmap(  # devices
        jax.vmap(  # update_batch_size
            jax.vmap(env.reset),  # n_envs
            axis_name="batch",
        ),
        axis_name="device",
    )(reset_keys)

    first_obs = first_timestep.observation
    first_done = first_timestep.last()[..., jnp.newaxis]  # ..., 1
    t0 = jnp.zeros((n_devices, cfg.system.update_batch_size), dtype=int)
    t0_train = jnp.zeros((n_devices, cfg.system.update_batch_size), dtype=int)

    # Initial learner state.
    learner_state = LearnerState(
        first_obs,
        first_done,
        init_hidden_state,
        env_state,
        t0,
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
) -> Any:  # TODO typing

    hidden_size = 256  # TODO dynamically get this

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
        actions = jax.vmap(select_single_random_action, (0, None, 0))(keys, action_options, p_vals)
        # unflatten to mtch expected action block size
        actions = jnp.reshape(actions, (batch_size, n_agents))

        return actions

    # INTERACT LEVEL 3 (1.2.2)
    def greedy(hidden_state, params, obs, done) -> Tuple[chex.Array, chex.Array]:
        """Uses online q values to greedily select the best action."""

        # Add dummy time dimension
        obs = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], obs)
        done = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], done)

        # Get q values
        new_hidden_state, q_values = q_net.apply(params, hidden_state, (obs, done))

        # Remove the dummy time dimension
        q_values = q_values[0]

        # Make unavailable actions quite negative
        q_vals_masked = jnp.where(
            obs.action_mask, q_values, jnp.zeros(q_values.shape) - 9999999
        )  # TODO: action masking in nw

        # Greedy argmax over action-value dim
        action = jnp.argmax(q_vals_masked, axis=-1)
        action = action[0]  # TODO check what dim this is... remove redundant first dim

        return action, new_hidden_state

    # INTERACT LEVEL 2 (1.2)
    def select_eps_greedy_action(
        action_selection_state: ActionSelectionState, obs: ObservationGlobalState, done: Array
    ) -> Tuple[ActionSelectionState, Array]:
        """Select action to take in eps-greedy way. Batch and agent dims are included."""

        def get_should_explore(key, t, shape):
            eps = jax.numpy.maximum(
                cfg.system.eps_min, 1 - (t / cfg.system.eps_decay) * (1 - cfg.system.eps_min)
            )
            return jax.random.uniform(key, shape) < eps

        # Unpacking
        params, hidden_state, t, key = action_selection_state

        # Random key splitting
        new_key, explore_key, selection_key = jax.random.split(key, 3)

        greedy_action, next_hidden_state = greedy(hidden_state, params, obs, done)

        action = jax.lax.select(
            get_should_explore(explore_key, t, greedy_action.shape),
            select_random_action_batch(selection_key, obs),
            greedy_action,
        )

        # repack new selection params
        next_action_selection_state = ActionSelectionState(
            params, next_hidden_state, t + cfg.arch.num_envs, new_key
        )  # TODO check t increment

        return next_action_selection_state, action

    # INTERACT LEVEL 1 (1.)
    def interaction_step(
        interaction_state: InteractionState, _: Any
    ) -> Tuple[InteractionState, Dict]:
        """Selects an action, steps global env, stores timesteps in global rb and repacks the parameters for the next step."""

        # light unpacking
        action_selection_state, env_state, buffer_state, obs, done = interaction_state

        # select the actions to take
        next_action_selection_state, action = select_eps_greedy_action(
            action_selection_state, obs, done
        )

        # step env with selected actions
        next_env_state, next_timestep = jax.vmap(env.step)(env_state, action)

        # Get reward
        reward = jnp.mean(
            next_timestep.reward, axis=-1, keepdims=True
        )  # NOTE: combine agent rewards

        transition = Transition(obs, action, reward, done)
        transition = jax.tree_util.tree_map(
            lambda x: x[:, jnp.newaxis, ...], transition
        )  # Add dummy time dim
        next_buffer_state = rb.add(buffer_state, transition)

        # Nexts
        next_obs = next_timestep.observation  # NB step!!
        next_done = next_timestep.last()[..., jnp.newaxis]

        new_interact_state = InteractionState(
            next_action_selection_state, next_env_state, next_buffer_state, next_obs, next_done
        )

        return new_interact_state, next_timestep.extras["episode_metrics"]

    # ___________________________________________________________________________________________________
    # TRAIN FUNCTIONS
    #
    # - 2. train (overarching)
    #    - 2.1 update_q
    #        - 1.2.1 q_loss_fn
    #        - 1.2.2 scan_apply
    #

    # TRAIN LEVEL 3 (2.1.1)
    def q_loss_fn(
        online_params: FrozenVariableDict, obs: Array, done: Array, action: Array, target: Array
    ) -> Tuple[Array, Metrics]:
        """The portion of the calculation to grad, namely online apply and mse with target."""
        q_online_params, online_mixer_params = online_params
        q_online = scan_apply(q_online_params, obs, done)
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

    # TRAIN LEVEL 3 and 4 (2.1.2, 2.1.1.1)
    def scan_apply(params, obs: ObservationGlobalState, done):
        """Applies RNN to a batch of trajectories by scanning over batch dim."""

        # B,T... -> T,B...
        def switch_leading_axis(arr: chex.Array):
            arr: Dict[str, chex.Array] = jax.tree_map(lambda x: jax.numpy.swapaxes(x, 0, 1), arr)
            return arr

        hidden_state = ScannedRNN.initialize_carry(
            (cfg.system.sample_batch_size, obs.agents_view.shape[2]), hidden_size
        )
        obs = switch_leading_axis(obs)  # B, T -> T, B
        done = switch_leading_axis(done)
        obs_done = (obs, done)

        _, next_q_vals_online = q_net.apply(params, hidden_state, obs_done)
        return switch_leading_axis(next_q_vals_online)  # swap time and batch again

    # TRAIN LEVEL 2 (2.1)
    def update_q(
        params: QMIXParams, opt_states: optax.OptState, data: Transition, t_train: int
    ) -> Tuple[QMIXParams, optax.OptState, Metrics]:
        """Update the Q parameters."""

        # Get the data associated with obs
        data_first: Dict[str, chex.Array] = jax.tree_map(lambda x: x[:, :-1, ...], data)
        # Get the data associated with next_obs
        data_next: Dict[str, chex.Array] = jax.tree_map(lambda x: x[:, 1:, ...], data)

        # next_term = data_next.term[...,jnp.newaxis]
        first_reward = data_first.reward
        next_done = data_next.done

        # Scan over each sample and discard first timestep
        next_q_vals_online = scan_apply(params.online, data.obs, data.done)
        next_q_vals_online = jnp.where(
            data.obs.action_mask, next_q_vals_online, jnp.zeros(next_q_vals_online.shape) - 9999999
        )[
            :, 1:, ...
        ]  # TODO: action masking in nw
        next_q_vals_target = scan_apply(params.target, data.obs, data.done)[:, 1:, ...]

        # Double q-value selection
        next_action = jnp.argmax(next_q_vals_online, axis=-1)
        next_q_val = jnp.squeeze(
            jnp.take_along_axis(next_q_vals_target, next_action[..., jnp.newaxis], axis=-1), axis=-1
        )
        next_q_val = mixer.apply(
            params.mixer_target, next_q_val, data_next.obs.global_state[:, :, 0, ...]
        )  # B,T,A,... -> B,T,1,...

        # TD Target
        target_q_val = (
            first_reward
            + (1.0 - jnp.array(next_done, dtype="float32")) * cfg.system.gamma * next_q_val
        )

        # Update Q function.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            (params.online, params.mixer_online),
            data_first.obs,
            data_first.done,
            data_first.action,
            target_q_val,
        )
        q_loss_info["mean_first_reward"] = jnp.mean(first_reward)
        q_loss_info["mean_next_qval"] = jnp.mean(next_q_val)
        q_loss_info["done"] = jnp.mean(data.done)

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
            next_online_params, next_target_params, next_mixer_params, next_mixer_target_params
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
        next_params, next_opt_states, q_loss_info = update_q(params, opt_states, data, t_train)

        next_train_state = TrainState(
            buffer_state, next_params, next_opt_states, t_train + 1, next_key
        )

        return next_train_state, q_loss_info

    # ___________________________________________________________________________________________________
    # INTERACT-TRAIN LOOP

    scanned_interact = lambda state: lax.scan(
        interaction_step, state, None, length=cfg.system.rollout_length
    )
    scanned_train = lambda state: lax.scan(train, state, None, length=cfg.system.epochs)

    # interact and train
    def update_step(
        learner_state: LearnerState, _: Any
    ) -> Tuple[LearnerState, Tuple[Metrics, Metrics]]:
        """Interact, then learn. The _ at the end of a var means updated."""

        # unpack and get random keys
        (
            obs,
            done,
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
        interaction_state = InteractionState(
            action_selection_state, env_state, buffer_state, obs, done
        )
        final_interaction_state, metrics = scanned_interact(interaction_state)

        # Sample and learn
        train_state = TrainState(
            final_interaction_state.buffer_state, params, opt_state, train_steps, train_key
        )
        final_train_state, losses = scanned_train(train_state)

        next_learner_state = LearnerState(
            final_interaction_state.obs,
            final_interaction_state.done,
            final_interaction_state.action_selection_state.hidden_state,
            final_interaction_state.env_state,
            final_interaction_state.action_selection_state.time_steps,
            final_train_state.train_steps,
            final_train_state.opt_state,
            final_interaction_state.buffer_state,
            final_train_state.params,
            new_key,
        )

        return next_learner_state, (metrics, losses)

    pmaped_steps = (
        1024  # todo: config option # NOTE (Claude) this is related to steps between logging.
    )

    pmaped_updated_step = jax.pmap(
        jax.vmap(
            lambda state: lax.scan(update_step, state, None, length=pmaped_steps),
            axis_name="batch",
        ),
        axis_name="device",
        donate_argnums=0,
    )

    return pmaped_updated_step, greedy


def run_experiment(cfg: DictConfig) -> float:
    n_devices = len(jax.devices())
    cfg.arch.n_devices = n_devices

    pmaped_steps = 1024  # todo: config option
    steps_per_rollout = n_devices * cfg.arch.num_envs * cfg.system.rollout_length * pmaped_steps
    max_episode_return = -jnp.inf

    (env, eval_env), learner_state, q_net, mixer, opts, rb, logger, key = init(cfg)
    update, greedy = make_update_fns(cfg, env, q_net, mixer, opts, rb)

    key, eval_key = jax.random.split(key)

    evaluator, absolute_metric_evaluator = make_eval_fns(
        eval_env=eval_env,
        network_apply_fn=q_net.apply,
        config=cfg,
        use_recurrent_net=True,
        scanned_rnn=ScannedRNN(),
    )

    # todo: don't need to return trained_params or eval keys
    # evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
    #     eval_env=eval_env,
    #     key=eval_key,
    #     network=nns, #NOTE (Louise) this part needs redoing with that replacement actor function
    #     params=learner_state.params.online,
    #     config=cfg,
    #     greedy=greedy
    # )

    if cfg.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=cfg,  # Save all config as metadata in the checkpoint
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.save_args,  # Checkpoint args
        )

    start_time = time.time()

    def get_epsilon(t: int):
        """Calculate epsilon for exploration rate using config variables."""
        eps = jax.numpy.maximum(
            cfg.system.eps_min, 1 - (t / cfg.system.eps_decay) * (1 - cfg.system.eps_min)
        )
        return eps

    # Main loop:
    # We want start to align with the final step of the first pmaped_learn,
    # where we've done explore_steps and 1 full learn step.
    start = steps_per_rollout
    for eval_idx, t in enumerate(range(start, int(cfg.system.total_timesteps), steps_per_rollout)):
        # Learn loop:
        learner_state, (metrics, losses) = update(learner_state)
        jax.block_until_ready(learner_state)

        # Log:
        # Multiply by learn steps here because anakin steps per second is learn + act steps
        # But we want to make sure we're counting env steps correctly so it's not included
        # in the loop counter.
        t_frm_learnstate = int(learner_state.time_steps[0][0].item())
        sps = t_frm_learnstate * cfg.system.epochs / (time.time() - start_time)
        final_metrics, _ = episode_metrics.get_final_step_metrics(metrics)
        loss_metrics = losses
        logger.log(
            {"step": t_frm_learnstate, "steps_per_second": sps},
            t_frm_learnstate,
            eval_idx,
            LogEvent.MISC,
        )
        logger.log(final_metrics, t_frm_learnstate, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t_frm_learnstate, eval_idx, LogEvent.TRAIN)
        logger.log(
            {"epsilon": get_epsilon(t_frm_learnstate)}, t_frm_learnstate, eval_idx, LogEvent.MISC
        )

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, cfg.arch.n_devices)
        eval_params = unreplicate_batch_dim(learner_state.params.online)
        eval_output = evaluator(eval_params, eval_keys)
        jax.block_until_ready(eval_output)

        # Log:
        episode_return = jnp.mean(eval_output.episode_metrics["episode_return"])
        logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.EVAL)

        # # Log:
        # episode_return = jnp.mean(eval_output.episode_metrics["episode_return"])
        # logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.EVAL)

        # # Save best actor params.
        # if cfg.arch.absolute_metric and max_episode_return <= episode_return:
        #     best_params = copy.deepcopy(unreplicate_batch_dim(learner_state.params.online))
        #     max_episode_return = episode_return

        # # Checkpoint:
        # if cfg.logger.checkpointing.save_model:
        #     # Save checkpoint of learner state
        #     unreplicated_learner_state = unreplicate_learner_state(learner_state)  # type: ignore
        #     checkpointer.save(
        #         timestep=t,
        #         unreplicated_learner_state=unreplicated_learner_state,
        #         episode_return=episode_return,
        #     )

    # # Measure absolute metric.
    # if cfg.arch.absolute_metric:
    #     eval_keys = jax.random.split(key, n_devices)

    #     eval_output = absolute_metric_evaluator(best_params, eval_keys)
    #     jax.block_until_ready(eval_output)

    #     logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    return float(episode_return)


@hydra.main(config_path="../../configs", config_name="default_rec_iql.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}QMIX experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()
