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
from typing import Any, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from rich.pretty import pprint

from mava import networks
from mava.evaluator import evaluator_setup
from mava.networks import ContinuousRecActor as Actor
from mava.networks import ScannedRNN
from mava.types import (
    ContRecActorApply,
    ExperimentOutput,
    HiddenStates,
    LearnerFn,
    OptStates,
    Params,
    RecCriticApply,
    RNNLearnerState,
    RNNPPOTransition,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax import unreplicate_learner_state
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.total_timestep_checker import check_total_timesteps
from mava.utils.training import (
    get_logprob_entropy,
    make_learning_rate,
    select_action_cont_ppo,
)


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ContRecActorApply, RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RNNLearnerState]:
    """Get the learner function."""

    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
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
                - dones (bool): Whether the last timestep was a terminal state.
                - hstates (HiddenStates): The current hidden states of the RNN.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, RNNPPOTransition]:
            """Step the environment."""
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                hstates,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, :], last_timestep.observation
            )
            ac_in = (
                batched_observation,
                last_done[:, 0][jnp.newaxis, :],
            )

            # Run the network.
            policy_hidden_state, actor_mean, actor_log_std = actor_apply_fn(
                params.actor_params, hstates.policy_hidden_state, ac_in
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, hstates.critic_hidden_state, ac_in
            )

            # Sample action from the policy and squeeze out the batch dimension.
            raw_action, action, log_prob = select_action_cont_ppo(
                (actor_mean, actor_log_std), policy_key, config.env.env_name
            )
            value, raw_action, action, log_prob = (
                value.squeeze(0),
                raw_action.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # log episode return and length
            done = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, config.system.num_agents).reshape(config.arch.num_envs, -1),
                timestep.last(),
            )
            info = timestep.extras["episode_metrics"]

            hstates = HiddenStates(policy_hidden_state, critic_hidden_state)
            transition = RNNPPOTransition(
                done,
                raw_action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                hstates,
                info,
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates
            )
            return learner_state, transition

        # INITIALISE RNN STATE
        initial_hstates = learner_state.hstates

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        (
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
        ) = learner_state

        # Add a batch dimension to the observation.
        batched_last_observation = jax.tree_util.tree_map(
            lambda x: x[jnp.newaxis, :], last_timestep.observation
        )
        ac_in = (
            batched_last_observation,
            last_done[:, 0][jnp.newaxis, :],
        )

        # Run the network.
        _, last_val = critic_apply_fn(params.critic_params, hstates.critic_hidden_state, ac_in)
        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)
        last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

        def _calculate_gae(
            traj_batch: RNNPPOTransition, last_val: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the GAE."""

            def _get_advantages(gae_and_next_value: Tuple, transition: RNNPPOTransition) -> Tuple:
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

        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                params, opt_states = train_state
                (
                    traj_batch,
                    advantages,
                    targets,
                ) = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    actor_opt_state: OptState,
                    traj_batch: RNNPPOTransition,
                    gae: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK

                    obs_and_done = (traj_batch.obs, traj_batch.done[:, :, 0])
                    _, actor_mean, actor_log_std = actor_apply_fn(
                        actor_params, traj_batch.hstates.policy_hidden_state[0], obs_and_done
                    )
                    log_prob, entropy = get_logprob_entropy(
                        actor_mean,
                        actor_log_std,
                        traj_batch.action,
                        config.env.env_name,
                    )

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

                    total_loss = loss_actor - config.system.ent_coef * entropy
                    return total_loss, (loss_actor, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    critic_opt_state: OptState,
                    traj_batch: RNNPPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    obs_and_done = (traj_batch.obs, traj_batch.done[:, :, 0])
                    _, value = critic_apply_fn(
                        critic_params, traj_batch.hstates.critic_hidden_state[0], obs_and_done
                    )

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = config.system.vf_coef * value_loss
                    return total_loss, (value_loss)

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

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="batch"
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

                return (new_params, new_opt_state), loss_info

            (
                params,
                opt_states,
                init_hstates,
                traj_batch,
                advantages,
                targets,
                key,
            ) = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            batch = (traj_batch, advantages, targets)
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            reshaped_batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (
                params,
                opt_states,
                init_hstates,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        init_hstates = jax.tree_util.tree_map(lambda x: x[None, :], initial_hstates)
        update_state = (
            params,
            opt_states,
            init_hstates,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, _, traj_batch, advantages, targets, key = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: RNNLearnerState) -> ExperimentOutput[RNNLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstateS (HiddenStates): The initial hidden states of the RNN.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RNNLearnerState], Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of actions and agents.
    config.system.num_agents = env.action_spec().shape[0]
    config.system.num_actions = env.action_spec().shape[1]

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimisers.
    actor_network, critic_network = networks.make(
        config=config, network="recurrent", centralised_critic=False
    )

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
    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(lambda x: x[0], init_obs)
    init_obs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs), dtype=bool)
    init_x = (init_obs, init_done)

    # Initialise hidden states.
    hidden_size = config.network.actor_network.pre_torso_layer_sizes[-1]
    init_policy_hstate = ScannedRNN.initialize_carry((config.arch.num_envs), hidden_size)
    init_critic_hstate = ScannedRNN.initialize_carry((config.arch.num_envs), hidden_size)

    # initialise params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_x)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Vmap network apply function over number of agents.
    vmapped_actor_network_apply_fn = jax.vmap(
        actor_network.apply, in_axes=(None, 1, (2, None)), out_axes=(1, 2, 2)
    )
    # Vmap network apply function over number of agents.
    vmapped_critic_network_apply_fn = jax.vmap(
        critic_network.apply, in_axes=(None, 1, (2, None)), out_axes=(1, 2)
    )

    # Get network apply functions and optimiser updates.
    apply_fns = (vmapped_actor_network_apply_fn, vmapped_critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Duplicate the hidden state for each agent.
    init_policy_hstate = jnp.expand_dims(init_policy_hstate, axis=1)
    init_policy_hstate = jnp.tile(init_policy_hstate, (1, config.system.num_agents, 1))

    init_critic_hstate = jnp.expand_dims(init_critic_hstate, axis=1)
    init_critic_hstate = jnp.tile(init_critic_hstate, (1, config.system.num_agents, 1))

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True
        )
        # Update the params and hstates
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_map(reshape_states, env_states)
    timesteps = jax.tree_map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, config.system.num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, hstates, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, hstates, step_keys, dones = replicate_learner
    init_learner_state = RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=hstates,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> None:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert (
            config.system.rollout_length % config.system.recurrent_chunk_size == 0
        ), "Rollout length must be divisible by recurrent chunk size."

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        network=actor_network,
        params=learner_state.params.actor_params,
        config=config,
        use_recurrent_net=True,
        scanned_rnn=ScannedRNN,
    )

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
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

    # Run experiment for a total number of evaluations.
    max_episode_return = jnp.float32(0.0)
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = steps_per_rollout * (eval_step + 1)
        learner_output.episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        logger.log(learner_output.episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = jax.tree_util.tree_map(
            lambda x: x[:, 0, ...], learner_output.learner_state.params.actor_params
        )
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        evaluator_output.episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_learner_state(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time

        t = steps_per_rollout * (eval_step + 1)
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()


@hydra.main(
    config_path="../configs", config_name="default_rec_ippo_continuous.yaml", version_base="1.2"
)
def hydra_entry_point(cfg: DictConfig) -> None:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent IPPO experiment completed{Style.RESET_ALL}")


if __name__ == "__main__":
    hydra_entry_point()
