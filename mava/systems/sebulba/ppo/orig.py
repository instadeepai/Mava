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

from mava.utils.sebulba_utils import configure_computation_environment

configure_computation_environment()  # noqa: E402

import copy
import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import PRNGKey
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import get_sebulba_ff_evaluator as evaluator_setup
from mava.logger import Logger
from mava.networks import get_networks
from mava.types import (
    ActorApply,
    CriticApply,
    LearnerState,
    Observation,
    OptStates,
    Params,
)
from mava.types import PPOTransition as Transition
from mava.types import SebulbaLearnerFn as LearnerFn
from mava.types import SingleDeviceFn
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax import merge_leading_dims
from mava.utils.make_env import make


def rollout(  # noqa: CCR001
    rng: PRNGKey,
    config: DictConfig,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    device_thread_id: int,
    apply_fns: Tuple,
    logger: Logger,
    learner_devices: List,
) -> None:
    """Executor rollout loop."""
    # Create envs
    envs = make(config)(config.arch.num_envs)  # type: ignore

    # Setup
    len_executor_device_ids = len(config.arch.executor_device_ids)
    t_env = 0
    start_time = time.time()

    # Get the apply functions for the actor and critic networks.
    vmap_actor_apply, vmap_critic_apply = apply_fns

    # Define the util functions: select action function and prepare data to share it with learner.
    @jax.jit
    def get_action_and_value(
        params: FrozenDict,
        observation: Observation,
        key: PRNGKey,
    ) -> Tuple:
        """Get action and value."""
        key, subkey = jax.random.split(key)

        policy = vmap_actor_apply(params.actor_params, observation)
        action, logprob = policy.sample_and_log_prob(seed=subkey)

        value = vmap_critic_apply(params.critic_params, observation).squeeze()
        return action, logprob, value, key

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        """Prepare data to share with learner."""
        return jax.tree_map(  # type: ignore
            lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage
        )

    # Define the episode info
    env_id = np.arange(config.arch.num_envs)
    # Accumulated episode returns
    episode_returns = np.zeros((config.arch.num_envs,), dtype=np.float32)
    # Final episode returns
    returned_episode_returns = np.zeros((config.arch.num_envs,), dtype=np.float32)
    # Accumulated episode lengths
    episode_lengths = np.zeros((config.arch.num_envs,), dtype=np.float32)
    # Final episode lengths
    returned_episode_lengths = np.zeros((config.arch.num_envs,), dtype=np.float32)

    # Define the data structure
    params_queue_get_time: deque = deque(maxlen=10)
    rollout_time: deque = deque(maxlen=10)
    rollout_queue_put_time: deque = deque(maxlen=10)

    # Reset envs
    next_obs, infos = envs.reset()
    next_dones = jnp.zeros((config.arch.num_envs, config.system.num_agents), dtype=jax.numpy.bool_)

    # Loop till the learner has finished training
    for update in range(1, config.system.num_updates + 2):
        # Setup
        env_recv_time: float = 0
        inference_time: float = 0
        storage_time: float = 0
        env_send_time: float = 0

        # Get the latest parameters from the learner
        params_queue_get_time_start = time.time()
        if config.arch.concurrency:
            if update != 2:
                params = params_queue.get()
                params.network_params["params"]["Dense_0"]["kernel"].block_until_ready()
        else:
            params = params_queue.get()
        params_queue_get_time.append(time.time() - params_queue_get_time_start)

        # Rollout
        rollout_time_start = time.time()
        storage: List = []
        # Loop over the rollout length
        for _ in range(0, config.system.rollout_length):
            # Get previous step info
            cached_next_obs = next_obs
            cached_next_dones = next_dones
            cashed_action_mask = np.stack(infos["actions_mask"])

            # Increment current timestep
            t_env += (
                config.arch.n_threads_per_executor * len_executor_device_ids * config.arch.num_envs
            )

            # Get action and value
            inference_time_start = time.time()
            (
                action,
                logprob,
                value,
                rng,
            ) = get_action_and_value(params, Observation(cached_next_obs, cashed_action_mask), rng)
            inference_time += time.time() - inference_time_start

            # Step the environment
            env_send_time_start = time.time()
            cpu_action = np.array(action)
            next_obs, next_reward, terminated, truncated, infos = envs.step(cpu_action)
            next_done = terminated + truncated
            next_dones = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, config.system.num_agents).reshape(config.arch.num_envs, -1),
                (next_done),
            )

            # Append data to storage
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()
            storage.append(
                Transition(
                    done=cached_next_dones,
                    action=action,
                    value=value,
                    reward=next_reward,
                    log_prob=logprob,
                    obs=cached_next_obs,
                    info=np.stack(infos["actions_mask"]),  # Add action mask to info
                )
            )
            storage_time += time.time() - storage_time_start

            # Update episode info
            episode_returns[env_id] += np.mean(next_reward)
            returned_episode_returns[env_id] = np.where(
                next_done,
                episode_returns[env_id],
                returned_episode_returns[env_id],
            )
            episode_returns[env_id] *= (1 - next_done) * (1 - truncated)
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                next_done,
                episode_lengths[env_id],
                returned_episode_lengths[env_id],
            )
            episode_lengths[env_id] *= (1 - next_done) * (1 - truncated)
        rollout_time.append(time.time() - rollout_time_start)

        # Prepare data to share with learner
        partitioned_storage = prepare_data(storage)
        sharded_storage = Transition(
            *list(  # noqa: C417
                map(
                    lambda x: jax.device_put_sharded(x, devices=learner_devices),  # type: ignore
                    partitioned_storage,
                )
            )
        )
        sharded_next_obs = jax.device_put_sharded(
            np.split(next_obs, len(learner_devices)), devices=learner_devices
        )
        sharded_next_done = jax.device_put_sharded(
            np.split(next_dones, len(learner_devices)), devices=learner_devices
        )
        sharded_next_action_mask = jax.device_put_sharded(
            np.split(np.stack(infos["actions_mask"]), len(learner_devices)), devices=learner_devices
        )
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

        if (update % config.arch.log_frequency == 0) or (config.system.num_updates + 1 == update):
            # Log info
            logger.log_executor_metrics(
                t_env=t_env,
                metrics={
                    "episodes_info": {
                        "episode_return": returned_episode_returns,
                        "episode_length": returned_episode_lengths,
                        "steps_per_second": int(t_env / (time.time() - start_time)),
                    },
                    "speed_info": {
                        "rollout_time": np.mean(rollout_time),
                    },
                    "queue_info": {
                        "params_queue_get_time": np.mean(params_queue_get_time),
                        "env_recv_time": env_recv_time,
                        "inference_time": inference_time,
                        "storage_time": storage_time,
                        "env_send_time": env_send_time,
                        "rollout_queue_put_time": np.mean(rollout_queue_put_time),
                    },
                },
                device_thread_id=device_thread_id,
            )


def get_learner_fn(
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn:
    """Get the learner function."""
    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def single_device_update(
        agents_state: LearnerState,
        traj_batch: Transition,
        last_observation: Observation,
        rng: PRNGKey,
    ) -> Tuple[LearnerState, chex.PRNGKey, Tuple]:
        params, opt_states, _, _, _ = agents_state

        def _calculate_gae(
            traj_batch: Transition, last_val: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the GAE."""

            def _get_advantages(gae_and_next_value: Tuple, transition: Transition) -> Tuple:
                """Calculate the GAE for a single transition."""
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                gamma = config.system.gamma
                delta = reward + gamma * next_value * (1 - done) - value
                gae = delta + gamma * config.system.gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # Calculate GAE
        last_val = critic_apply_fn(params.critic_params, last_observation)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    actor_opt_state: OptStates,
                    traj_batch: Transition,
                    gae: chex.Array,
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
                    entropy = actor_policy.entropy().mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    return total_loss_actor, (loss_actor, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    critic_opt_state: OptStates,
                    traj_batch: Transition,
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
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params, opt_states.actor_opt_state, traj_batch, advantages
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                critic_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, opt_states.critic_opt_state, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the learner devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="local_devices"
                )
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="local_devices"
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
                loss_info = (total_loss, value_loss, actor_loss, entropy)

                return (new_params, new_opt_state), loss_info

            params, opt_states, traj_batch, advantages, targets, rng = update_state
            rng, shuffle_rng = jax.random.split(rng)

            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_rng, batch_size)
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
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, rng)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, rng)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, rng = update_state
        learner_state = agents_state._replace(params=params, opt_states=opt_states)
        return learner_state, rng, loss_info

    def learner_fn(
        agents_state: LearnerState,
        sharded_storages: List,
        sharded_next_obs: List,
        sharded_next_done: List,
        sharded_next_action_mask: List,
        key: chex.PRNGKey,
    ) -> Tuple:
        """Single device update."""
        # Horizontal stack all the data from different devices
        traj_batch = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
        traj_batch = traj_batch._replace(obs=Observation(traj_batch.obs, traj_batch.info))

        # Get last observation
        last_obs = jnp.concatenate(sharded_next_obs)
        last_action_mask = jnp.concatenate(sharded_next_action_mask)
        last_observation = Observation(last_obs, last_action_mask)

        # Update learner
        agents_state, key, (total_loss, value_loss, actor_loss, entropy) = single_device_update(
            agents_state, traj_batch, last_observation, key
        )

        # Pack loss info
        loss_info = {
            "total_loss": total_loss,
            "loss_actor": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }
        return agents_state, key, loss_info

    return learner_fn


def learner_setup(
    rngs: chex.Array, config: DictConfig, learner_devices: List
) -> Tuple[SingleDeviceFn, LearnerState, Tuple[ActorApply, ActorApply]]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get number of actions and agents.
    dummy_envs = make(config)(  # type: ignore
        config.arch.num_envs  # Create dummy_envs to get observation and action spaces
    )
    config.system.num_agents = dummy_envs.single_observation_space.shape[0]
    config.system.num_actions = int(dummy_envs.single_action_space.nvec[0])

    # PRNG keys.
    actor_net_key, critic_net_key = rngs

    # Define network and optimiser.
    actor_network, critic_network = get_networks(
        config=config, network="feedforward", centralised_critic=False
    )
    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(config.system.actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(config.system.critic_lr, eps=1e-5),
    )

    # Initialise observation: Select only obs for a single agent.
    init_obs = np.array([dummy_envs.single_observation_space.sample()[0]])
    init_action_mask = np.ones((1, config.system.num_actions))
    init_x = Observation(init_obs, init_action_mask)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Vmap network apply function over number of agents.
    vmapped_actor_network_apply_fn = jax.vmap(
        actor_network.apply,
        in_axes=(None, Observation(1, 1, None)),
        out_axes=(1),
    )
    vmapped_critic_network_apply_fn = jax.vmap(
        critic_network.apply,
        in_axes=(None, Observation(1, 1, None)),
        out_axes=(1),
    )

    # Pack apply and update functions.
    apply_fns = (vmapped_actor_network_apply_fn, vmapped_critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Define agents state
    agents_state = LearnerState(
        params=Params(
            actor_params=actor_params,
            critic_params=critic_params,
        ),
        opt_states=OptStates(
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
        ),
    )
    # Replicate agents state per learner device
    agents_state = flax.jax_utils.replicate(agents_state, devices=learner_devices)

    # Get Learner function: pmap over learner devices.
    single_device_update = get_learner_fn(apply_fns, update_fns, config)
    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=learner_devices,
    )

    # Close dummy envs.
    dummy_envs.close()

    return multi_device_update, agents_state, apply_fns


def run_experiment(_config: DictConfig) -> None:  # noqa: CCR001
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Setup device distribution.
    local_devices = jax.local_devices() #why are we using local devices insted of devices? ------------------------------------------------------------------------------------------------------------------------------------ define a ratio insted of the devices to use?
    learner_devices = [local_devices[d_id] for d_id in config.arch.learner_device_ids]

    # PRNG keys.
    rng, rng_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )
    learner_keys = jax.device_put_replicated(rng, learner_devices)

    # Sanity check of config
    assert (
        config.arch.num_envs % len(config.arch.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)" 
    #each thread is going to devide needs to give an equal number of traj to each learning device? shound't each actor Thread have a designated N learneres? If we have less actor T than learners then ech actor will devide based on the num_env and gives to N actors, ig to lessen the managment each actor gives to all of the learners? 
    #this deviates from the paper? 
    assert (
        int(config.arch.num_envs / len(config.arch.learner_device_ids))
        * config.arch.n_threads_per_executor
        % config.system.num_minibatches
        == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches" #this one makes sense but the assertion is a bit off?

    # Setup learner.
    (
        multi_device_update,
        agents_state,
        apply_fns,
    ) = learner_setup((actor_net_key, critic_net_key), config, learner_devices)

    # Setup evaluator.
    eval_envs = make(config)(config.arch.num_eval_episodes)  # type: ignore
    evaluator = evaluator_setup(eval_envs=eval_envs, apply_fn=apply_fns[0], config=config)

    # Calculate total timesteps.
    batch_size = int(
        config.arch.num_envs
        * config.system.rollout_length
        * config.arch.n_threads_per_executor
        * len(config.arch.executor_device_ids)
    )
    config.system.total_timesteps = config.system.num_updates * batch_size

    # Setup logger.
    config.arch.log_frequency = config.system.num_updates // config.arch.num_evaluation
    logger = Logger(config)
    cfg_dict: Dict = OmegaConf.to_container(config, resolve=True)
    pprint(cfg_dict)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=cfg_dict,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    if config.logger.checkpointing.load_model:
        print(
            f"{Fore.RED}{Style.BRIGHT}Loading checkpoint is not supported\
            for sebulba architecture yet{Style.RESET_ALL}"
        )

    # Executor setup and launch.
    unreplicated_params = flax.jax_utils.unreplicate(agents_state.params)
    params_queues: List = []
    rollout_queues: List = []
    for d_idx, d_id in enumerate(  # Loop through each executor device
        config.arch.executor_device_ids
    ):
        # Replicate params per executor device
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        # Loop through each executor thread
        for thread_id in range(config.arch.n_threads_per_executor):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            params_queues[-1].put(device_params)
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(rng, local_devices[d_id]),
                    config,
                    rollout_queues[-1],
                    params_queues[-1],
                    d_idx * config.arch.n_threads_per_executor + thread_id,
                    apply_fns,
                    logger,
                    learner_devices,
                ),
            ).start()

    # Run experiment for the total number of updates.
    rollout_queue_get_time: deque = deque(maxlen=10)
    data_transfer_time: deque = deque(maxlen=10)
    trainer_update_number = 0
    max_episode_return = jnp.float32(0.0)
    best_params = None
    while True:
        trainer_update_number += 1
        rollout_queue_get_time_start = time.time()
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

        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()

        # Update learner
        (agents_state, learner_keys, loss_info) = multi_device_update(  # type: ignore
            agents_state,
            sharded_storages,
            sharded_next_obss,
            sharded_next_dones,
            sharded_next_action_masks,
            learner_keys,
        )

        # Send updated params to executors
        unreplicated_params = flax.jax_utils.unreplicate(agents_state.params)
        for d_idx, d_id in enumerate(config.arch.executor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            for thread_id in range(config.arch.n_threads_per_executor):
                params_queues[d_idx * config.arch.n_threads_per_executor + thread_id].put(
                    device_params
                )

        if trainer_update_number % config.arch.log_frequency == 0:
            # Logging training info
            logger.log_trainer_metrics(
                experiment_output={
                    "loss_info": loss_info,
                    "queue_info": {
                        "rollout_queue_get_time": np.mean(rollout_queue_get_time),
                        "data_transfer_time": np.mean(data_transfer_time),
                        "rollout_params_queue_get_time_diff": np.mean(rollout_queue_get_time)
                        - avg_params_queue_get_time,
                        "rollout_queue_size": rollout_queues[0].qsize(),
                        "params_queue_size": params_queues[0].qsize(),
                    },
                    "speed_info": {
                        "training_time": time.time() - training_time_start,
                        "trainer_update_number": trainer_update_number,
                    },
                },
                t_env=t_env,
            )

            # Evaluation
            rng_e, _ = jax.random.split(rng_e)
            evaluator_output = evaluator(params=unreplicated_params, rng=rng_e)
            # Log the results of the evaluation.
            episode_return = logger.log_evaluator_metrics(
                t_env=t_env,
                metrics=evaluator_output,
                eval_step=trainer_update_number,
            )

            if save_checkpoint:
                # Save checkpoint of learner state
                checkpointer.save(
                    timestep=t_env,
                    unreplicated_learner_state=flax.jax_utils.unreplicate(agents_state),
                    episode_return=episode_return,
                )

            if config.arch.absolute_metric and max_episode_return <= episode_return:
                best_params = copy.deepcopy(unreplicated_params)
                max_episode_return = episode_return

        # Check if training is finished
        if trainer_update_number >= config.system.num_updates:
            rng_e, _ = jax.random.split(rng_e)
            # Measure absolute metric
            evaluator_output = evaluator(params=best_params, rng=rng_e, eval_multiplier=10)
            # Log the results of the evaluation.
            logger.log_evaluator_metrics(
                t_env=t_env,
                metrics=evaluator_output,
                eval_step=trainer_update_number + 1,
                absolute_metric=True,
            )
            break


@hydra.main(config_path="../../configs", config_name="default_ff_ippo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> None:
    """Experiment entry point."""

    # Run experiment.
    run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")


if __name__ == "__main__":
    hydra_entry_point()