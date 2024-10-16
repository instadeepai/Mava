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
from functools import partial
from typing import Any, Callable, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict as Params
from jax import tree
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import get_eval_fn, get_num_eval_envs, make_sable_act_fn
from mava.networks import SableNetwork
from mava.systems.sable.types import (
    ExecutionApply,
    HiddenStates,
    LearnerState,
    TrainingApply,
    Transition,
)
from mava.types import ExperimentOutput, LearnerFn, MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.sable_utils import concat_time_and_agents, get_init_hidden_state
from mava.utils.total_timestep_checker import check_total_timesteps
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ExecutionApply, TrainingApply],
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""

    # Get apply functions for executing and training the network.
    sable_action_select_fn, sable_apply_fn = apply_fns

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (FrozenDict): The current model parameters.
                - opt_states (OptState): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - hstates (HiddenStates): The hidden state of the network.
            _ (Any): The current metrics info.

        """

        def _env_step(learner_state: LearnerState, _: int) -> Tuple[LearnerState, Transition]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep, hstates = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)

            # Apply the actor network to get the action, log_prob, value and updated hstates.
            last_obs = last_timestep.observation
            action, log_prob, value, hstates = sable_action_select_fn(  # type: ignore
                params,
                last_obs.agents_view,
                last_obs.action_mask,
                hstates,
                policy_key,
            )
            action = jnp.squeeze(action, axis=-1)
            log_prob = jnp.squeeze(log_prob, axis=-1)
            value = jnp.squeeze(value, axis=-1)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            info = tree.map(
                lambda x: jnp.repeat(x[..., jnp.newaxis], config.system.num_agents, axis=-1),
                timestep.extras["episode_metrics"],
            )

            # Reset hidden state if done.
            done = timestep.last()
            done = jnp.expand_dims(done, (1, 2, 3, 4))
            hstates = tree.map(lambda hs: jnp.where(done, jnp.zeros_like(hs), hs), hstates)

            # SET TRANSITION
            prev_done = tree.map(
                lambda x: jnp.repeat(x, config.system.num_agents).reshape(config.arch.num_envs, -1),
                last_timestep.last(),
            )
            transition = Transition(
                prev_done,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                info,
            )
            learner_state = LearnerState(params, opt_states, key, env_state, timestep, hstates)
            return learner_state, transition

        # COPY OLD HIDDEN STATES: TO BE USED IN THE TRAINING LOOP
        prev_hstates = tree.map(lambda x: jnp.copy(x), learner_state.hidden_state)

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step,
            learner_state,
            jnp.arange(config.system.rollout_length),
            config.system.rollout_length,
        )

        # CALCULATE ADVANTAGE
        params, opt_states, key, env_state, last_timestep, updated_hstates = learner_state
        key, last_val_key = jax.random.split(key)
        _, _, current_val, _ = sable_action_select_fn(  # type: ignore
            params,
            last_timestep.observation.agents_view,
            last_timestep.observation.action_mask,
            updated_hstates,
            last_val_key,
        )
        current_val = jnp.squeeze(current_val, axis=-1)
        current_done = tree.map(
            lambda x: jnp.repeat(x, config.system.num_agents).reshape(config.arch.num_envs, -1),
            last_timestep.last(),
        )

        def _calculate_gae(
            traj_batch: Transition,
            current_val: chex.Array,
            current_done: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the GAE."""

            def _get_advantages(
                carry: Tuple[chex.Array, chex.Array, chex.Array], transition: Transition
            ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
                """Calculate the GAE for a single transition."""
                gae, next_value, next_done = carry
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                gamma = config.system.gamma
                delta = reward + gamma * next_value * (1 - next_done) - value
                gae = delta + gamma * config.system.gae_lambda * (1 - next_done) * gae
                return (gae, value, done), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(current_val), current_val, current_done),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, current_val, current_done)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_state = train_state
                traj_batch, advantages, targets, prev_hstates = batch_info

                def _loss_fn(
                    params: Params,
                    traj_batch: Transition,
                    gae: chex.Array,
                    value_targets: chex.Array,
                    prev_hstates: HiddenStates,
                ) -> Tuple:
                    """Calculate Sable loss."""
                    # RERUN NETWORK
                    log_prob, value, entropy = sable_apply_fn(  # type: ignore
                        params,
                        traj_batch.obs.agents_view,
                        traj_batch.action,
                        traj_batch.obs.action_mask,
                        prev_hstates,
                        traj_batch.done,
                    )
                    log_prob = jnp.squeeze(log_prob, axis=-1)
                    value = jnp.squeeze(value, axis=-1)
                    entropy = jnp.squeeze(entropy, axis=-1)

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
                    entropy = entropy.mean()

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )

                    # MSE LOSS
                    value_losses = jnp.square(value - value_targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    # TOTAL LOSS
                    total_loss = (
                        loss_actor
                        - config.system.ent_coef * entropy
                        + config.system.vf_coef * value_loss
                    )
                    return total_loss, (loss_actor, entropy, value_loss)

                # CALCULATE ACTOR LOSS
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                loss_info, grads = grad_fn(
                    params,
                    traj_batch,
                    advantages,
                    targets,
                    prev_hstates,
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="batch")
                # pmean over devices.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="device")

                # UPDATE PARAMS AND OPTIMISER STATE
                updates, new_opt_state = update_fn(grads, opt_state)
                new_params = optax.apply_updates(params, updates)

                # PACK LOSS INFO
                total_loss = loss_info[0]
                actor_loss = loss_info[1][0]
                entropy = loss_info[1][1]
                value_loss = loss_info[1][2]
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
                traj_batch,
                advantages,
                targets,
                key,
                prev_hstates,
            ) = update_state

            # SHUFFLE MINIBATCHES
            key, batch_shuffle_key, agent_shuffle_key = jax.random.split(key, 3)

            # Shuffle batch
            batch_size = config.arch.num_envs
            batch_perm = jax.random.permutation(batch_shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: jnp.take(x, batch_perm, axis=1), batch)
            # Shuffle agents
            agent_perm = jax.random.permutation(agent_shuffle_key, config.system.num_agents)
            batch = jax.tree_util.tree_map(lambda x: jnp.take(x, agent_perm, axis=2), batch)

            # Shuffle hidden states
            prev_hstates = jax.tree_util.tree_map(
                lambda x: jnp.take(x, batch_perm, axis=0), prev_hstates
            )

            # CONCATENATE TIME AND AGENTS
            batch = jax.tree_util.tree_map(concat_time_and_agents, batch)

            # SPLIT INTO MINIBATCHES
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                batch,
            )
            prev_hs_minibatch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                prev_hstates,
            )

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch,
                (params, opt_states),
                (*minibatches, prev_hs_minibatch),
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
                prev_hstates,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
            prev_hstates,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key, _ = update_state
        learner_state = LearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            updated_hstates,
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (FrozenDict): The initial model parameters.
                - opt_state (OptState): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - hstates (HiddenStates): The initial hidden states of the network.

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
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Callable, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, net_key = keys

    # Get number of agents and actions.
    action_dim = int(env.action_spec().num_values[0])
    n_agents = env.action_spec().shape[0]
    config.system.num_agents = n_agents
    config.system.num_actions = action_dim

    # Define network.
    sable_network = SableNetwork(
        n_block=config.network.n_block,
        embed_dim=config.network.embed_dim,
        n_head=config.network.n_head,
        n_agents=n_agents,
        action_dim=action_dim,
        decay_scaling_factor=config.network.decay_scaling_factor,
        action_space_type="discrete",
    )

    # Define optimiser.
    lr = make_learning_rate(config.system.actor_lr, config)
    optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    # Get mock inputs to initialise network.
    init_obs = env.observation_spec().generate_value()
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)  # Add batch dim
    init_action = jnp.zeros((1, n_agents), dtype=jnp.int32)
    init_hs = get_init_hidden_state(config.network, config.arch.num_envs)
    init_hs = tree.map(lambda x: x[0, jnp.newaxis], init_hs)
    init_dones = jnp.zeros((1, n_agents), dtype=bool)

    # Initialise params and optimiser state.
    params = sable_network.init(
        net_key,
        init_obs.agents_view,
        init_action,
        init_obs.action_mask,
        init_hs,
        init_dones,
    )
    opt_state = optim.init(params)

    # Pack apply and update functions.
    apply_fns = (
        partial(sable_network.apply, method="get_actions"),  # Execution function
        sable_network.apply,  # Training function
    )
    update_fn = optim.update

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fn, config)
    learn = jax.pmap(learn, axis_name="device")

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
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Initialise hidden state.
    init_hstates = get_init_hidden_state(config.network, config.arch.num_envs)

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
    replicate_learner = (params, opt_state, step_keys)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)
    init_hstates = tree.map(broadcast, init_hstates)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())
    init_hstates = flax.jax_utils.replicate(init_hstates, devices=jax.devices())

    # Initialise learner state.
    params, opt_state, step_keys = replicate_learner

    init_learner_state = LearnerState(
        params=params,
        opt_states=opt_state,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        hidden_state=init_hstates,
    )

    return learn, apply_fns[0], init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config)

    # PRNG keys.
    key, key_e, net_key = jax.random.split(jax.random.PRNGKey(config.system.seed), num=3)

    # Setup learner.
    learn, sable_execution_fn, learner_state = learner_setup(env, (key, net_key), config)

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_sable_act_fn(sable_execution_fn)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
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

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = get_init_hidden_state(config.network, eval_batch_size)
    eval_hs = tree.map(lambda x: x[jnp.newaxis], eval_hs)

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {"hidden_state": eval_hs})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        abs_hs = get_init_hidden_state(config.network, eval_batch_size)
        abs_hs = tree.map(lambda x: x[jnp.newaxis], abs_hs)
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": abs_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_sable.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "rec_sable"

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Sable Memory experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
