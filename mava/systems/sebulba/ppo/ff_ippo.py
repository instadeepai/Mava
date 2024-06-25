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
from typing import Any, Dict, Tuple, List
import threading
import chex
import flax
import hydra
import jax
import jax.debug
import jax.numpy as jnp
import numpy as np
import optax
import queue
from collections import deque
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from rich.pretty import pprint

from mava.evaluator import make_sebulba_eval_fns as make_eval_fns #todo: make a standered eval function 
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.sebulba.ppo.types import LearnerState, OptStates, Params, PPOTransition, Observation #todo: change this Observation to use the origial one 
from mava.types import ActorApply, CriticApply, ExperimentOutput, LearnerFn
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.total_timestep_checker import sebulba_check_total_timesteps
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


def rollout(    
    key: chex.PRNGKey,
    config: DictConfig,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    device_thread_id: int,
    apply_fns: Tuple,
    logger: MavaLogger,
    learner_devices: List,
    actor_device_id : int):
    
    #create envs
    env = environments.make_gym_env(config, config.arch.num_envs)
    
    #setup
    len_executor_device_ids = len(config.arch.executor_device_ids)
    current_actor_device = jax.devices()[actor_device_id]
    t_env = 0
    start_time = time.time()

    actor_apply_fn, critic_apply_fn = apply_fns
    
    # Define the util functions: select action function and prepare data to share it with learner.
    @jax.jit
    def get_action_and_value(
        params: FrozenDict,
        observation: Observation,
        key: chex.PRNGKey,
    ) -> Tuple:
        """Get action and value."""
        key, subkey = jax.random.split(key)
        
        actor_policy = actor_apply_fn(params.actor_params, observation)
        action = actor_policy.sample(seed=subkey)
        log_prob = actor_policy.log_prob(action)

        value = critic_apply_fn(params.critic_params, observation).squeeze()
        return action, log_prob, value, key

    @jax.jit
    def prepare_data(storage: List[PPOTransition]) -> PPOTransition:
        """Prepare data to share with learner."""
        return jax.tree_map(  # type: ignore
            lambda *xs : jnp.stack(xs), *storage
        )


    # Define queues to track time
    params_queue_get_time: deque = deque(maxlen=10)
    rollout_time: deque = deque(maxlen=10)
    rollout_queue_put_time: deque = deque(maxlen=10)
    
    next_obs , info = env.reset() #todo : the first info is discarded , is that a problem?
    next_dones = jnp.zeros((config.arch.num_envs, config.system.num_agents), dtype=jax.numpy.bool_)
    
    move_to_device = lambda x : jax.device_put(x, device = current_actor_device)

    # Loop till the learner has finished training
    for eval_step in range(config.arch.num_evaluation):
        for update in range(1, config.system.num_updates_per_eval + 2):
            # Setup
            env_recv_time: float = 0
            inference_time: float = 0
            storage_time: float = 0
            env_send_time: float = 0
            
            # Get the latest parameters from the learner
            params_queue_get_time_start = time.time()
            params = params_queue.get()
            params_queue_get_time.append(time.time() - params_queue_get_time_start)
            
            # Rollout   
            rollout_time_start = time.time()
            storage: List = []
            # Loop over the rollout length
            for _ in range(0, config.system.rollout_length):
                # Cached for transition
                cached_next_obs = move_to_device(jnp.stack(next_obs, axis = 1))
                cached_next_dones = move_to_device(next_dones)
                cashed_action_mask = move_to_device(jnp.stack([*info["actions_mask"]], axis = 0) ) #unpack the numpy object, find a more pythonic way?
                
                # Increment current timestep
                t_env += (
                    config.arch.n_threads_per_executor * len_executor_device_ids * config.arch.num_envs
                )
                
                # Get action and value
                inference_time_start = time.time()
                #
                (
                    action,
                    log_prob,
                    value,
                    key,
                ) = get_action_and_value(params, Observation(cached_next_obs, cashed_action_mask), key)
                inference_time += time.time() - inference_time_start
                
                # Step the environment
                env_send_time_start = time.time()
                cpu_action = jax.device_get(action)
                next_obs, next_reward, terminated, truncated, info = env.step(cpu_action.swapaxes(0,1)) #num_env, num_agents --> num_agents, num_env 
                next_dones = np.logical_or(terminated, truncated)         
                
                metrics = jax.tree_map(lambda *x : jnp.asarray(x), *info["metrics"]) # Stack the metrics (N_envs , N_metrics) -- > (N_metrics, N_envs)
                # Append data to storage
                env_send_time += time.time() - env_send_time_start
                storage_time_start = time.time()
                storage.append(
                    PPOTransition(
                        done=cached_next_dones,
                        action=action,
                        value=value,
                        reward=next_reward,
                        log_prob=log_prob,
                        obs=Observation(cached_next_obs, cashed_action_mask),
                        info=metrics, 
                        )#todo: use a threadsafe alt https://github.com/instadeepai/CityLearn/blob/27e69f8ebdf1789c55ffab5c326bfaa50733a5e7/power_systems/sax_sebulba.py#L39
                )
                storage_time += time.time() - storage_time_start

            rollout_time.append(time.time() - rollout_time_start) 
        
        # Prepare data to share with learner 
        # todo: investigate te thread --> single learning  
        partitioned_storage = prepare_data(storage)
        #sorage has shape rollout_len, num_agents, num_envs, .... while the other vectors have num_agents, num_envs, ... -> their split axis is diffrent
        shard_split_payload= lambda x, axis : jax.device_put_sharded(jnp.split(x, len(learner_devices), axis=axis), devices=learner_devices)

        sharded_storage = jax.tree_map(lambda x : shard_split_payload(x, 1) , partitioned_storage)
        
        sharded_next_obs = shard_split_payload(jnp.stack(next_obs, axis = 1), 0) 
        sharded_next_action_mask = shard_split_payload(jnp.stack([*info["actions_mask"]], axis = 0), 0)  
        sharded_next_done = shard_split_payload(next_dones, 0)

        payload = (
            t_env,
            sharded_storage,
            sharded_next_obs,
            sharded_next_done,
            sharded_next_action_mask,
            np.mean(params_queue_get_time),
        )

        # Put data in the rollout queue to share it with the learner
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)

            
def get_learner_fn(
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: LearnerState, _: Any, traj_batch : PPOTransition, last_obs: chex.Array, last_action_mask : chex.Array, last_dones : chex.Array) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.
        """

        def _calculate_gae( #todo: lake sure this is appropriate 
            traj_batch: PPOTransition, last_val: chex.Array, last_done: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            def _get_advantages(
                carry: Tuple[chex.Array, chex.Array, chex.Array], transition: PPOTransition
            ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
                gae, next_value, next_done = carry
                done, value, reward = transition.done, transition.value, transition.reward
                gamma = config.system.gamma
                delta = reward + gamma * next_value * (1 - next_done) - value
                gae = delta + gamma * config.system.gae_lambda * (1 - next_done) * gae
                return (gae, value, done), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val, last_done),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value
        
        # CALCULATE ADVANTAGE
        params, opt_states, key, _, _ = learner_state
        last_val = critic_apply_fn(params.critic_params, Observation(last_obs, last_action_mask))
        advantages, targets = _calculate_gae(traj_batch, last_val, last_dones)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    actor_opt_state: OptState,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    return total_loss_actor, (loss_actor, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    critic_opt_state: OptState,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    value = critic_apply_fn(critic_params, traj_batch.obs)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    critic_total_loss = config.system.vf_coef * value_loss
                    return critic_total_loss, (value_loss)

                # CALCULATE ACTOR LOSS
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params,
                    opt_states.actor_opt_state,
                    traj_batch,
                    advantages,
                    entropy_key,
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                critic_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, opt_states.critic_opt_state, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                # pmean over devices.
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="device"
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)
                # PACK LOSS INFO
                total_loss = actor_loss_info[0] + critic_loss_info[0]
                value_loss = critic_loss_info[1]
                actor_loss = actor_loss_info[1][0]
                entropy = actor_loss_info[1][1]
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)
            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs * len(config.arch.executor_device_ids) * config.arch.n_threads_per_executor
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            # UPDATE MINIBATCHES
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)
        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = LearnerState(params, opt_states, key, None, None)
        metric = traj_batch.info #todo: metrci calcualtions
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: LearnerState, traj_batch : PPOTransition, last_obs: chex.Array, last_action_mask : chex.Array, last_dones : chex.Array) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
        """
        # Broadcast static parameters for scan
        partial_update_step =  lambda learner_state,  xs :  _update_step(learner_state,  xs, traj_batch , last_obs, last_action_mask, last_dones) 
        
        learner_state, (episode_info, loss_info) = jax.lax.scan(
            partial_update_step, learner_state, None, config.system.num_updates_per_eval
        )

        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    keys: chex.Array, config: DictConfig, learner_devices: List
) -> Tuple[LearnerFn[LearnerState], Actor, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(learner_devices)
    
    #create temporory envoirnments.
    env  = environments.make_gym_env(config, config.arch.num_envs)
    # Get number of agents and actions.
    action_space = env.single_action_space
    config.system.num_agents = len(action_space)
    config.system.num_actions = action_space[0].n 

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.action_head, action_dim=config.system.num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso)

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation: Select only obs for a single agent.
    init_obs = jnp.array([env.single_observation_space.sample()])
    init_action_mask = jnp.ones((config.system.num_agents, config.system.num_actions))
    init_x = Observation(init_obs, init_action_mask)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = Params(actor_params, critic_params)

    # Pack apply and update functions.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device", devices = learner_devices)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, step_keys)

    # Duplicate learner across Learner devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=learner_devices)

    # Initialise learner state.
    params, opt_states, step_keys = replicate_learner
    init_learner_state = LearnerState(params, opt_states, step_keys, None, None)
    env.close()

    return learn, apply_fns, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    devices = jax.devices() # todo: use local devices insted? 
    learner_devices = [devices[d_id] for d_id in config.arch.learner_device_ids]

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    learner_keys = jax.device_put_replicated(key, learner_devices)
    
    # Sanity check of config
    assert (
        config.arch.num_envs % len(config.arch.learner_device_ids) == 0
    ), "The number of environments must to be divisible by the number of learners "  
    
    assert (
        int(config.arch.num_envs / len(config.arch.learner_device_ids))
        * config.arch.n_threads_per_executor
        % config.system.num_minibatches
        == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches" 

    
    # Setup learner.
    learn, apply_fns , learner_state = learner_setup(
        (key ,actor_net_key, critic_net_key), config, learner_devices
    )

    # Setup evaluator.
    # One key per device for evaluation.
    evaluator, absolute_metric_evaluator = make_eval_fns(environments.make_gym_env, apply_fns[0], config) #todo: make this more generic

    # Calculate total timesteps.
    config = sebulba_check_total_timesteps(config) #todo: update this for sebulba
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        len(config.arch.executor_device_ids)
        * config.arch.n_threads_per_executor
        * config.system.rollout_length
        * config.arch.num_envs
        * config.system.num_updates_per_eval
    )

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )
        
    # Executor setup and launch.
    unreplicated_params = flax.jax_utils.unreplicate(learner_state.params)
    params_queues: List = []
    rollout_queues: List = []
    for d_idx, d_id in enumerate(  # Loop through each executor device
        config.arch.executor_device_ids
    ):
        # Replicate params per executor device
        device_params = jax.device_put(unreplicated_params, devices[d_id])
        # Loop through each executor thread
        for thread_id in range(config.arch.n_threads_per_executor):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            params_queues[-1].put(device_params)
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(key, devices[d_id]),
                    config,
                    rollout_queues[-1],
                    params_queues[-1],
                    d_idx * config.arch.n_threads_per_executor + thread_id,
                    apply_fns,
                    logger,
                    learner_devices,
                    d_id,
                ),
            ).start()
            
    # Run experiment for the total number of updates.
    rollout_queue_get_time: deque = deque(maxlen=10)
    data_transfer_time: deque = deque(maxlen=10)
    trainer_update_number = 0
    max_episode_return = jnp.float32(0.0)
    best_params = None
    for eval_step in range(config.arch.num_evaluation): #todo : place holder
        trainer_update_number += 1
        start_time = time.time()
        sharded_storages = []
        sharded_next_obss = []
        sharded_next_dones = []
        sharded_next_action_masks = []

        # Loop through each executor device
        for d_idx, _ in enumerate(config.arch.executor_device_ids):
            # Loop through each executor thread
            for thread_id in range(config.arch.n_threads_per_executor):
                # Get data from rollout queue
                (
                    t_env,
                    sharded_storage,
                    sharded_next_obs,
                    sharded_next_done,
                    sharded_next_action_mask,
                    avg_params_queue_get_time,
                ) = rollout_queues[d_idx * config.arch.n_threads_per_executor + thread_id].get()
                sharded_storages.append(sharded_storage)
                sharded_next_obss.append(sharded_next_obs)
                sharded_next_dones.append(sharded_next_done)
                sharded_next_action_masks.append(sharded_next_action_mask)
        rollout_queue_get_time.append(time.time() - start_time)
        training_time_start = time.time()
        
        #Concatinate the returned trajectories on the n_env axis 
        sharded_storages = jax.tree_map(lambda *x : jnp.concatenate(x, axis = 2), *sharded_storages)  #todo: check if this breaks the explicet array device placment 
        sharded_next_obss = jnp.concatenate(sharded_next_obss, axis = 1)
        sharded_next_dones = jnp.concatenate(sharded_next_dones, axis = 1)
        sharded_next_action_masks = jnp.concatenate(sharded_next_action_masks, axis = 1)

        learner_output = learn(learner_state, sharded_storages, sharded_next_obss, sharded_next_action_masks, sharded_next_dones)
        
        # Send updated params to executors
        unreplicated_params = flax.jax_utils.unreplicate(learner_output.learner_state.params)
        for d_idx, d_id in enumerate(config.arch.executor_device_ids):
            device_params = jax.device_put(unreplicated_params, devices[d_id])
            for thread_id in range(config.arch.n_threads_per_executor):
                params_queues[d_idx * config.arch.n_threads_per_executor + thread_id].put(
                    device_params
                )
                
        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics) # todo: these shapes are not as expected  
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
                
        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Evaluation on the learner 
        key_e, eval_key = jax.random.split(key_e, 2)
        episode_metrics = evaluator(unreplicate_n_dims(learner_output.learner_state.params.actor_params, 1 ), eval_key)
        
        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(episode_metrics["episode_length"]))
        episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(episode_metrics, t, eval_step, LogEvent.EVAL)
        
        #todo: add saving
        
        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(learner_output.learner_state.params)
            max_episode_return = episode_return
    #todo: abs metric
    return None#eval_performance


@hydra.main(config_path="../../../configs", config_name="default_ff_ippo_seb.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()

#learner_output.episode_metrics.keys()
#dict_keys(['episode_length', 'episode_return'])