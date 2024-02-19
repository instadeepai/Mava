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
import time
from typing import Any, Callable, Dict, NamedTuple, Tuple

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
from jax import Array
from jax.typing import ArrayLike
from jumanji.env import Environment, State
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypeAlias

from mava.evaluator_idqn import evaluator_setup
from mava.types import Observation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax import unreplicate_batch_dim, unreplicate_learner_state
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import episode_metrics

# jax.config.update("jax_platform_name", "cpu")


# todo: types.py
class Qs(NamedTuple):
    online: FrozenVariableDict
    target: FrozenVariableDict

class QLearnParams(NamedTuple):
    dqns:Qs
    # NOTE (Louise) later add Qmix network

class LearnerState(NamedTuple):
    obs: Array
    env_state: State
    buffer_state: TrajectoryBuffer
    params: QLearnParams
    opt_state: optax.OptState
    t: Array
    key: PRNGKey

class Transition(NamedTuple):
    obs: Array
    action: Array
    reward: Array
    done: Array
    next_obs: Array


Metrics = Dict[str, Array]
BufferState: TypeAlias = TrajectoryBufferState[Transition]
Networks: TypeAlias = Tuple[nn.Module, nn.Module]
Optimisers: TypeAlias = Tuple[
    optax.GradientTransformation
]

class QNetwork(nn.Module):
    num_actions:int
    def setup(self) -> None:
        self.net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu, nn.Dense(self.num_actions)])


    @nn.compact
    def __call__(self, obs: Observation) -> Array:
        x = obs.agents_view
        return self.net(x)


def init(
    cfg: DictConfig,
) -> Tuple[
    Tuple[Environment, Environment],
    Networks,
    Optimisers,
    TrajectoryBuffer,
    LearnerState,
    Array,
    MavaLogger,
    chex.PRNGKey,
]:
    logger = MavaLogger(cfg)

    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()
    n_devices = len(devices)

    def replicate(x: Any) -> Any:
        """First replicate the update batch dim then put on devices."""
        x = jax.tree_map(lambda y: jnp.broadcast_to(y, (cfg.system.update_batch_size, *y.shape)), x)
        return jax.device_put_replicated(x, devices)

    env, eval_env = environments.make(cfg)

    num_actions = int(env.action_spec().num_values[0]) # NOTE (Claude) got this from the PPO systems

    key, q_key, q_target_key = jax.random.split(key, 3)

    init_obs = env.observation_spec().generate_value()
    init_acts = env.action_spec().generate_value()
    init_obs_batched = jax.tree_map(lambda x: x[0][jnp.newaxis, ...], init_obs)

    # Making Q networks
    q = QNetwork(num_actions)
    q_params = q.init(q_key, init_obs_batched)
    q_target_params = q.init(q_target_key, init_obs_batched)

    # Pack params
    params = QLearnParams(Qs(q_params, q_target_params))

    q_opt = optax.adam(cfg.system.q_lr)
    q_opt_state = q_opt.init(params.dqns.online)


    # Distribute params and opt states across all devices
    params = replicate(params)
    q_opt_state = replicate(q_opt_state)

    # Create replay buffer
    init_transition = Transition(
        obs=init_obs,
        action=init_acts,
        # todo: n agents rewards/discounts
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=init_obs,
    )

    rb = fbx.make_item_buffer(
        max_length=cfg.system.buffer_size,
        min_length=cfg.system.buffer_min_size,
        sample_batch_size=cfg.system.batch_size,
        add_batches=True,
    )
    buffer_state = replicate(rb.init(init_transition))

    nns = (q, q) # NOTE (Louise) replace a q with mixer params later

    # Reset env.
    n_keys = cfg.system.n_envs * n_devices * cfg.system.update_batch_size
    key_shape = (n_devices, cfg.system.update_batch_size, cfg.system.n_envs, -1)
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

    t = jnp.zeros((n_devices, cfg.system.update_batch_size), dtype=int)

    # Initial learner state.
    learner_state = LearnerState(
        first_obs, env_state, buffer_state, params, q_opt_state, t, first_keys
    )
    return (env, eval_env), nns, q_opt, rb, learner_state, logger, key 


def make_update_fns(
    cfg: DictConfig,
    env: Environment,
    nns: Networks,
    opts: Optimisers,
    rb: TrajectoryBuffer,
    # target_entropy: chex.Array,
) -> Tuple[
    Callable[[LearnerState], Tuple[LearnerState, Metrics]],
    Callable[[LearnerState], Tuple[LearnerState, Tuple[Metrics, Metrics]]],
]:
    _, q = nns # NOTE (Louise) put mixer later
    q_opt = opts

    # full_action_shape = (cfg.system.n_envs, *env.action_spec().shape) # NOTE (Claude) I dont think this is right since we have discrete actions

    def step(
        action: Array, obs: Observation, env_state: State, buffer_state: BufferState
    ) -> Tuple[Array, State, BufferState, Dict]:
        """Given an action, step the environment and add to the buffer."""
        env_state, timestep = jax.vmap(env.step)(env_state, action)
        next_obs = timestep.observation
        rewards = timestep.reward[:, 0]
        terms = ~(timestep.discount).astype(bool)[:, 0]
        infos = timestep.extras

        real_next_obs = infos["real_next_obs"]

        transition = Transition(obs, action, rewards, terms, real_next_obs)
        buffer_state = rb.add(buffer_state, transition)

        return next_obs, env_state, buffer_state, infos["episode_metrics"]

    # losses:
    def q_loss_fn(q_online_params: FrozenVariableDict, obs: Array, action: Array, target: Array) -> Tuple[Array, Metrics]:
        q_online = q.apply(q_online_params, obs)
        q_loss = jnp.mean((q_online - target) ** 2)

        loss_info = {
            "q_loss": q_loss,
            "mean_q": jnp.mean(q_online),
        }

        return q_loss, loss_info

    # Update functions:
    def update_q(
        params: QLearnParams, opt_states: optax.OptState, data: Transition, key: chex.PRNGKey
    ) -> Tuple[QLearnParams, optax.OptState, Metrics]:
        """Update the Q parameters."""
        # Calculate Q target values.
        rewards = data.reward[..., jnp.newaxis, jnp.newaxis]
        dones = data.done[..., jnp.newaxis, jnp.newaxis]

        next_q_vals_online = q.apply(params.dqns.online, data.next_obs) #NOTE (Louise) not superduper efficient
        next_q_vals_target = q.apply(params.dqns.target, data.next_obs)

        # TODO (Claude) do double q-value selection here...
        next_action = jnp.argmax(next_q_vals_online, axis=-1)
        next_q_val = jnp.take(next_q_vals_target, next_action[...,jnp.newaxis])

        target_q_val = (rewards + (1.0 - dones) * cfg.system.gamma * next_q_val)

        # Update Q function.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(params.dqns.online, data.obs, data.action, target_q_val)
        # Mean over the device and batch dimension.
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_updates, new_q_opt_state = q_opt.update(q_grads, opt_states)
        new_online_q_params = optax.apply_updates(params.dqns.online, q_updates)

        # Target network polyak update.
        new_target_q_params = optax.incremental_update(
            new_online_q_params, params.dqns.target, cfg.system.tau
        )

        # Repack params and opt_states.
        q_and_target = Qs(new_online_q_params, new_target_q_params)
        params = params._replace(dqns=q_and_target)
        #opt_states = opt_states._replace(q=new_q_opt_state)

        return params, new_q_opt_state, q_loss_info

    # Act/learn loops:
    def update_epoch(
        carry: Tuple[BufferState, QLearnParams, optax.OptState, int, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[BufferState, QLearnParams, optax.OptState, int, chex.PRNGKey], Metrics]:
        """Update the Q function and optionally policy/alpha with TD3 delayed update."""
        buffer_state, params, opt_states, t, key = carry
        key, buff_key, q_key, actor_key = jax.random.split(key, 4)

        # sample
        data = rb.sample(buffer_state, buff_key).experience

        # learn
        params, opt_states, q_loss_info = update_q(params, opt_states, data, q_key)

        return (buffer_state, params, opt_states, t, key), q_loss_info
    

    def get_epsilon(t:int):
        """Calculate epsilon for exploration rate using config hyperparameters."""
        eps = jax.numpy.maximum(
            cfg.system.eps_min,
            1-(t/cfg.system.eps_decay)*(1-cfg.system.eps_min)
            )
        return eps
    
    # Using probability values allows for non-ragged masking - must be done per row because of "choice".
    def select_single_action(key, action_options, p_values):
        """Select one action per row with probabilities given by p_values."""
        action = jax.random.choice(key,action_options,p=p_values)
        return action
    
    def select_random_action_batch(key, action_mask):
        """Select actions randomly with masking."""
        batch_size, n_agents, n_actions = action_mask.shape

        keys = jax.random.split(key,batch_size*n_agents)
        action_options = jnp.arange(n_actions)

        num_avail_per_agent = jnp.sum(action_mask,axis=-1)[...,jnp.newaxis]

        p_vals = action_mask.astype("int32")/num_avail_per_agent
        p_vals = jnp.reshape(p_vals,(batch_size*n_agents,n_actions))

        actions = jax.vmap(select_single_action, (0,None,0))(keys,action_options,p_vals)
        actions = jnp.reshape(actions,(batch_size,n_agents))#,1))

        return actions

    def greedy(online_params:FrozenVariableDict, obs:Array):
        # get q values
        q_values = q.apply(online_params, obs)
        # make unavailable actions quite negative
        q_vals_masked = jnp.where(obs.action_mask,q_values,jnp.zeros(q_values.shape)-1000) #make more general
        # greedy argmax
        action = jnp.argmax(q_vals_masked, axis=-1)#.reshape(q_values.shape[0],q_values.shape[1],1)
        return action
    
    

    def select_eps_greedy_action(online_params:FrozenVariableDict, obs:Array, t:int, key:chex.PRNGKey):
        """Select action to take in eps-greedy way. Batch and agent dims are included."""
        # get exploration rate
        epsilon = get_epsilon(t)

        # decide whether to explore
        key,key_1 = jax.random.split(key,2)
        should_explore = jax.random.uniform(key)<epsilon 

        # choose actions accordingly
        action = jax.lax.select(
            should_explore,
            select_random_action_batch(key_1, obs.action_mask), #is it general enough?
            greedy(online_params, obs)
        )

        return action

    def act( 
        carry: Tuple[FrozenVariableDict, Array, State, BufferState, int, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[FrozenVariableDict, Array, State, BufferState, int, chex.PRNGKey], Dict]:
        """Acting loop: select action, step env, add to buffer."""
        online_params, obs, env_state, buffer_state, t, key = carry

        action = jnp.zeros((128,3),dtype="int32")

        #action = select_eps_greedy_action(online_params,obs,t,key)

        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)

        return (online_params, next_obs, env_state, buffer_state, t , key), metrics


    scanned_update = lambda state: lax.scan(update_epoch, state, None, length=cfg.system.epochs)
    scanned_act = lambda state: lax.scan(act, state, None, length=cfg.system.rollout_length)

    # Act loop -> sample -> update loop
    def update_step(carry: LearnerState, _: Any) -> Tuple[LearnerState, Tuple[Metrics, Metrics]]:
        """Act, sample, learn."""

        obs, env_state, buffer_state, params, opt_states, t, key = carry
        key, act_key, learn_key = jax.random.split(key, 3)
        # Act
        act_state = (params.dqns.online, obs, env_state, buffer_state, t, act_key)
        (_, next_obs, env_state, buffer_state, t, _), metrics = scanned_act(act_state)

        # Sample and learn
        learn_state = (buffer_state, params, opt_states, t, learn_key)
        (buffer_state, params, opt_states, _, _), losses = scanned_update(learn_state)

        t += cfg.system.n_envs * cfg.system.rollout_length
        return (
            LearnerState(next_obs, env_state, buffer_state, params, opt_states, t, key),
            (metrics, losses),
        )

    
    pmaped_steps = 1024  # todo: config option # NOTE (Claude) this is related to steps between logging.


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

    pmaped_steps = 1024  # todo: config option
    steps_per_rollout = n_devices * cfg.system.n_envs * cfg.system.rollout_length * pmaped_steps
    max_episode_return = -jnp.inf

    (env, eval_env), nns, opts, rb, learner_state, logger, key = init(cfg)
    update, greedy = make_update_fns(cfg, env, nns, opts, rb)

    _,q = nns
    key, eval_key = jax.random.split(key)
    # todo: don't need to return trained_params or eval keys
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key=eval_key,
        network=q, #NOTE (Louise) this part needs redoing with that replacement actor function
        params=learner_state.params.dqns.online,
        config=cfg,
        greedy=greedy
    )

    if cfg.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=cfg,  # Save all config as metadata in the checkpoint
            model_name=cfg.logger.system_name,
            **cfg.logger.checkpointing.save_args,  # Checkpoint args
        )

    start_time = time.time()


    def get_epsilon(t:int):
        """Calculate epsilon for exploration rate using config variables."""
        eps = jax.numpy.maximum(
            cfg.system.eps_min,
            1-(t/cfg.system.eps_decay)*(1-cfg.system.eps_min)
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

        sps = t * cfg.system.epochs / (time.time() - start_time)
        final_metrics = episode_metrics.get_final_step_metrics(metrics)
        loss_metrics = losses #| {"log_alpha": learner_state.params.log_alpha}
        logger.log({"step": t, "steps_per_second": sps}, t, eval_idx, LogEvent.MISC)
        logger.log(final_metrics, t, eval_idx, LogEvent.ACT)
        logger.log(loss_metrics, t, eval_idx, LogEvent.TRAIN)
        logger.log({"epsilon":get_epsilon(t)}, t, eval_idx, LogEvent.MISC)

        # Evaluate:
        key, eval_key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, n_devices)
        # todo: bug likely here -> don't have batch vmap yet so shouldn't unreplicate_batch_dim
        eval_output = evaluator(unreplicate_batch_dim(learner_state.params.dqns.online), eval_keys)
        jax.block_until_ready(eval_output)

        # Log:
        episode_return = jnp.mean(eval_output.episode_metrics["episode_return"])
        logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.EVAL)

        # Save best actor params.
        if cfg.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(unreplicate_batch_dim(learner_state.params.dqns.online))
            max_episode_return = episode_return

        # Checkpoint:
        if cfg.logger.checkpointing.save_model:
            # Save checkpoint of learner state
            unreplicated_learner_state = unreplicate_learner_state(learner_state)  # type: ignore
            checkpointer.save(
                timestep=t,
                unreplicated_learner_state=unreplicated_learner_state,
                episode_return=episode_return,
            )

    # Measure absolute metric.
    if cfg.arch.absolute_metric:
        eval_keys = jax.random.split(key, n_devices)

        eval_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(eval_output)

        logger.log(eval_output.episode_metrics, t, eval_idx, LogEvent.ABSOLUTE)

    logger.stop()

    return float(max_episode_return)


@hydra.main(config_path="../configs", config_name="default_ff_idqn.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}IDQN experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()
