# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

"""Trainer components for gradient step calculations."""
import abc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import jax
import jax.numpy as jnp
import optax
import reverb
import tree
from acme.jax import utils
from jax import jit

import mava.components.jax.building.adders  # To avoid circular imports
import mava.components.jax.training.model_updating  # To avoid circular imports
from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.building.datasets import TrainerDataset, TrajectoryDataset
from mava.components.jax.building.loggers import Logger
from mava.components.jax.building.networks import Networks
from mava.components.jax.building.parameter_client import TrainerParameterClient
from mava.components.jax.training.advantage_estimation import GAE
from mava.components.jax.training.base import Batch, TrainingState
from mava.components.jax.training.trainer import BaseTrainerInit
from mava.core_jax import SystemTrainer


@dataclass
class TrainerStepConfig:
    random_key: int = 42


class TrainerStep(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: TrainerStepConfig = TrainerStepConfig(),
    ):
        """Defines the hooks to override to step the trainer."""
        self.config = config

    @abc.abstractmethod
    def on_training_step(self, trainer: SystemTrainer) -> None:
        """Do a training step and log the results."""
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "step"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        TrajectoryDataset required to set up trainer.store.dataset_iterator.
        Step required to set up trainer.store.step_fn.
        TrainerParameterClient required to set up trainer.store.trainer_parameter_client
        and trainer.store.trainer_counts.
        Logger required to set up trainer.store.trainer_logger.

        Returns:
            List of required component classes.
        """
        return [TrajectoryDataset, Step, TrainerParameterClient, Logger]


class DefaultTrainerStep(TrainerStep):
    def __init__(
        self,
        config: TrainerStepConfig = TrainerStepConfig(),
    ):
        """Component defines the default trainer step.

        Sample -> execute step function -> sync parameter client
        -> update counts -> log.

        Args:
            config: TrainerStepConfig.
        """
        self.config = config

    def on_training_step(self, trainer: SystemTrainer) -> None:
        """Does a step of SGD and logs the results.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        # Do a batch of SGD.
        sample = next(trainer.store.dataset_iterator)

        # print("Sampled exprience: ", sample)
        # import tensorflow as tf
        # tf.print("Sampled exprience: ", sample)
        # exit()

        results = trainer.store.step_fn(sample)

        # Update our counts and record it.
        # counts = self._counter.increment(steps=1) # TODO: add back in later

        # TODO (dries): Confirm that this is the correctly place to put the
        # variable client code.
        timestamp = time.time()
        elapsed_time = (
            timestamp - trainer.store.timestamp
            if hasattr(trainer.store, "timestamp")
            else 0
        )
        trainer.store.timestamp = timestamp

        trainer.store.trainer_parameter_client.add_async(
            {"trainer_steps": 1, "trainer_walltime": elapsed_time},
        )

        # Update the variable source and the trainer.
        trainer.store.trainer_parameter_client.set_and_get_async()

        # Add the trainer counts.
        results.update(trainer.store.trainer_counts)

        # Write to the loggers.
        trainer.store.trainer_logger.write({**results})


class Step(Component):
    @abc.abstractmethod
    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """[summary]"""

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "sgd_step"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        TrainerDataset required for config sample_batch_size.
        BaseTrainerInit required to set up trainer.store.networks
        and trainer.store.trainer_agent_net_keys.
        Networks required to set up trainer.store.base_key.

        Returns:
            List of required component classes.
        """
        return [
            TrainerDataset,
            BaseTrainerInit,
            Networks,
        ]


@dataclass
class MAPGWithTrustRegionStepConfig:
    discount: float = 0.99


class MAPGWithTrustRegionStep(Step):
    def __init__(
        self,
        config: MAPGWithTrustRegionStepConfig = MAPGWithTrustRegionStepConfig(),
    ):
        """Component defines the MAPGWithTrustRegion SGD step.

        Args:
            config: MAPGWithTrustRegionStepConfig.
        """
        self.config = config

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        """Compute and store full batch size.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """
        # Note (dries): Assuming the batch and sequence dimensions are flattened.
        trainer.store.full_batch_size = (
            trainer.store.global_config.sample_batch_size
            * (trainer.store.global_config.sequence_length - 1)
        )

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """Define and store the SGD step function for MAPGWithTrustRegion.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        @jit
        def sgd_step(
            states: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step.

            Args:
                states: Training states (network params and optimiser states).
                sample: Reverb sample.

            Returns:
                Tuple[new state, metrics].
            """

            # Extract the data.
            data = sample.data

            observations, actions, rewards, termination, extra = (
                data.observations,
                data.actions,
                data.rewards,
                data.discounts,
                data.extras,
            )

            discounts = tree.map_structure(
                lambda x: x * self.config.discount, termination
            )

            behavior_log_probs = extra["policy_info"]

            networks = trainer.store.networks["networks"]

            def get_behavior_values(
                net_key: Any, reward: Any, observation: Any
            ) -> jnp.ndarray:
                """Gets behaviour values from the agent networks and observations."""
                o = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])), observation
                )
                behavior_values = networks[net_key].critic_network.apply(
                    states.critic_params[net_key], o
                )
                behavior_values = jnp.reshape(behavior_values, reward.shape[0:2])
                return behavior_values

            # TODO (Ruan): Double check this
            agent_nets = trainer.store.trainer_agent_net_keys
            behavior_values = {
                key: get_behavior_values(
                    agent_nets[key], rewards[key], observations[key].observation
                )
                for key in agent_nets.keys()
            }

            # Vmap over batch dimension
            batch_gae_advantages = jax.vmap(trainer.store.gae_fn, in_axes=0)

            advantages = {}
            target_values = {}
            for key in rewards.keys():
                advantages[key], target_values[key] = batch_gae_advantages(
                    rewards[key], discounts[key], behavior_values[key]
                )

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_sequences, num_steps, ..]
            (
                observations,
                actions,
                behavior_log_probs,
                behavior_values,
            ) = jax.tree_util.tree_map(
                lambda x: x[:, :-1],
                (observations, actions, behavior_log_probs, behavior_values),
            )

            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                behavior_log_probs=behavior_log_probs,
                target_values=target_values,
                behavior_values=behavior_values,
            )

            # Concatenate all trajectories. Reshape from [num_sequences, num_steps,..]
            # to [num_sequences * num_steps,..]
            agent_0_t_vals = list(target_values.values())[0]
            assert len(agent_0_t_vals) > 1
            num_sequences = agent_0_t_vals.shape[0]
            num_steps = agent_0_t_vals.shape[1]
            batch_size = num_sequences * num_steps
            assert batch_size % trainer.store.global_config.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.global_config.num_minibatches)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            (
                new_key,
                new_policy_params,
                new_critic_params,
                new_policy_opt_states,
                new_critic_opt_states,
                _,
            ), metrics = jax.lax.scan(
                trainer.store.epoch_update_fn,
                (
                    states.random_key,
                    states.policy_params,
                    states.critic_params,
                    states.policy_opt_states,
                    states.critic_opt_states,
                    batch,
                ),
                (),
                length=trainer.store.global_config.num_epochs,
            )

            # Set the metrics
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics["norm_policy_params"] = optax.global_norm(states.policy_params)
            metrics["norm_critic_params"] = optax.global_norm(states.critic_params)
            metrics["observations_mean"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_util.tree_map(
                        lambda x: jnp.abs(jnp.mean(x, axis=(0, 1))), observations
                    ),
                    num_batch_dims=0,
                )
            )
            metrics["observations_std"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_util.tree_map(
                        lambda x: jnp.std(x, axis=(0, 1)), observations
                    ),
                    num_batch_dims=0,
                )
            )
            metrics["rewards_mean"] = jax.tree_util.tree_map(
                lambda x: jnp.mean(jnp.abs(jnp.mean(x, axis=(0, 1)))), rewards
            )
            metrics["rewards_std"] = jax.tree_util.tree_map(
                lambda x: jnp.std(x, axis=(0, 1)), rewards
            )

            new_states = TrainingState(
                policy_params=new_policy_params,
                critic_params=new_critic_params,
                policy_opt_states=new_policy_opt_states,
                critic_opt_states=new_critic_opt_states,
                random_key=new_key,
            )
            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:
            """Step over the reverb sample and update the parameters / optimiser states.

            Args:
                sample: Reverb sample.

            Returns:
                Metrics from SGD step.
            """

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks["networks"]
            policy_params = {
                net_key: networks[net_key].policy_params for net_key in networks.keys()
            }
            critic_params = {
                net_key: networks[net_key].critic_params for net_key in networks.keys()
            }
            policy_opt_states = trainer.store.policy_opt_states
            critic_opt_states = trainer.store.critic_opt_states
            random_key, _ = jax.random.split(trainer.store.base_key)

            states = TrainingState(
                policy_params=policy_params,
                critic_params=critic_params,
                policy_opt_states=policy_opt_states,
                critic_opt_states=critic_opt_states,
                random_key=random_key,
            )

            new_states, metrics = sgd_step(states, sample)

            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimizer and random_key to the variable
            # server.
            trainer.store.base_key = new_states.random_key

            # These updates must remain separate for loops since the policy and critic
            # networks could have different layers.
            networks = trainer.store.networks["networks"]

            policy_params = {
                net_key: networks[net_key].policy_params for net_key in networks.keys()
            }
            for net_key in policy_params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks["networks"][net_key].policy_params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.policy_params[net_key][param_key]

                # Update the policy optimizer
                # This needs to be in the loop to not lose the reference.
                trainer.store.policy_opt_states[net_key] = new_states.policy_opt_states[
                    net_key
                ]

            critic_params = {
                net_key: networks[net_key].critic_params for net_key in networks.keys()
            }
            for net_key in critic_params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks["networks"][net_key].critic_params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.critic_params[net_key][param_key]

                # Update the policy optimizer
                # This needs to be in the loop to not lose the reference.
                trainer.store.critic_opt_states[net_key] = new_states.critic_opt_states[
                    net_key
                ]

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        GAE required to set up trainer.store.gae_fn.
        MAPGEpochUpdate required for config num_epochs, num_minibatches,
        and trainer.store.epoch_update_fn.
        MinibatchUpdate required to set up trainer.store.opt_states.
        ParallelSequenceAdder required for config sequence_length.

        Returns:
            List of required component classes.
        """
        return Step.required_components() + [
            GAE,
            mava.components.jax.training.model_updating.MAPGEpochUpdate,
            mava.components.jax.training.model_updating.MinibatchUpdate,
            mava.components.jax.building.adders.ParallelSequenceAdder,
        ]
