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
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import reverb
import tree
from acme.jax import utils
from jax import jit

from mava.components.jax import Component
from mava.components.jax.training import (
    Batch,
    Step,
    TrainingState,
    TrainingStateSeparateNetworks,
)
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
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    @abc.abstractmethod
    def on_training_step(self, trainer: SystemTrainer) -> None:
        """Do a training step and log the results."""
        pass

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "step"


class DefaultTrainerStep(TrainerStep):
    def __init__(
        self,
        config: TrainerStepConfig = TrainerStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_step(self, trainer: SystemTrainer) -> None:
        """Does a step of SGD and logs the results."""

        # Do a batch of SGD.
        sample = next(trainer.store.dataset_iterator)
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


@dataclass
class MAPGWithTrustRegionStepConfig:
    discount: float = 0.99


class MAPGWithTrustRegionStep(Step):
    def __init__(
        self,
        config: MAPGWithTrustRegionStepConfig = MAPGWithTrustRegionStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        # Note (dries): Assuming the batch and sequence dimensions are flattened.
        trainer.store.full_batch_size = trainer.store.sample_batch_size * (
            trainer.store.sequence_length - 1
        )

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        @jit
        def sgd_step(
            states: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

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
                o = jax.tree_map(
                    lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])), observation
                )
                _, behavior_values = networks[net_key].network.apply(
                    states.params[net_key], o
                )
                behavior_values = jnp.reshape(behavior_values, reward.shape[0:2])
                return behavior_values

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
            observations, actions, behavior_log_probs, behavior_values = jax.tree_map(
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
            assert batch_size % trainer.store.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.num_minibatches)
            batch = jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            (new_key, new_params, new_opt_states, _,), metrics = jax.lax.scan(
                trainer.store.epoch_update_fn,
                (states.random_key, states.params, states.opt_states, batch),
                (),
                length=trainer.store.num_epochs,
            )

            # Set the metrics
            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_params"] = optax.global_norm(states.params)
            metrics["observations_mean"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(
                        lambda x: jnp.abs(jnp.mean(x, axis=(0, 1))), observations
                    ),
                    num_batch_dims=0,
                )
            )
            metrics["observations_std"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(lambda x: jnp.std(x, axis=(0, 1)), observations),
                    num_batch_dims=0,
                )
            )
            metrics["rewards_mean"] = jax.tree_map(
                lambda x: jnp.mean(jnp.abs(jnp.mean(x, axis=(0, 1)))), rewards
            )
            metrics["rewards_std"] = jax.tree_map(
                lambda x: jnp.std(x, axis=(0, 1)), rewards
            )

            new_states = TrainingState(
                params=new_params, opt_states=new_opt_states, random_key=new_key
            )
            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            opt_states = trainer.store.opt_states
            random_key, _ = jax.random.split(trainer.store.key)

            states = TrainingState(
                params=params, opt_states=opt_states, random_key=random_key
            )

            new_states, metrics = sgd_step(states, sample)

            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimizer and random_key to the variable
            # server.
            trainer.store.key = new_states.random_key

            networks = trainer.store.networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            for net_key in params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks["networks"][net_key].params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.params[net_key][param_key]

                # Update the optimizer
                # This needs to be in the loop to not lose the reference.
                trainer.store.opt_states[net_key] = new_states.opt_states[net_key]

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGWithTrustRegionStepConfig


######################################
# SEPARATE POLICY AND CRITIC NETWORKS
######################################


class MAPGWithTrustRegionStepSeparateNetworks(Step):
    def __init__(
        self,
        config: MAPGWithTrustRegionStepConfig = MAPGWithTrustRegionStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        # Note (dries): Assuming the batch and sequence dimensions are flattened.
        trainer.store.full_batch_size = trainer.store.sample_batch_size * (
            trainer.store.sequence_length - 1
        )

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        @jit
        def sgd_step(
            states: TrainingStateSeparateNetworks, sample: reverb.ReplaySample
        ) -> Tuple[TrainingStateSeparateNetworks, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

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
                o = jax.tree_map(
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
            observations, actions, behavior_log_probs, behavior_values = jax.tree_map(
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
            assert batch_size % trainer.store.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.num_minibatches)
            batch = jax.tree_map(
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
                length=trainer.store.num_epochs,
            )

            # Set the metrics
            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_policy_params"] = optax.global_norm(states.policy_params)
            metrics["norm_critic_params"] = optax.global_norm(states.critic_params)
            metrics["observations_mean"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(
                        lambda x: jnp.abs(jnp.mean(x, axis=(0, 1))), observations
                    ),
                    num_batch_dims=0,
                )
            )
            metrics["observations_std"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(lambda x: jnp.std(x, axis=(0, 1)), observations),
                    num_batch_dims=0,
                )
            )
            metrics["rewards_mean"] = jax.tree_map(
                lambda x: jnp.mean(jnp.abs(jnp.mean(x, axis=(0, 1)))), rewards
            )
            metrics["rewards_std"] = jax.tree_map(
                lambda x: jnp.std(x, axis=(0, 1)), rewards
            )

            new_states = TrainingStateSeparateNetworks(
                policy_params=new_policy_params,
                critic_params=new_critic_params,
                policy_opt_states=new_policy_opt_states,
                critic_opt_states=new_critic_opt_states,
                random_key=new_key,
            )
            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:

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
            random_key, _ = jax.random.split(trainer.store.key)

            states = TrainingStateSeparateNetworks(
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
            trainer.store.key = new_states.random_key

            # TODO (Ruan): this could probably be done in one loop
            # instead of a different policy and critic loop to save time
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
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGWithTrustRegionStepConfig


######################################
# MAPPO
######################################
@dataclass
class MAPPOStepConfig:
    discount: float = 0.99


class MAPPOStepSeparateNetworks(Step):
    def __init__(
        self,
        config: MAPPOStepConfig = MAPPOStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        # Note (dries): Assuming the batch and sequence dimensions are flattened.
        trainer.store.full_batch_size = trainer.store.sample_batch_size * (
            trainer.store.sequence_length - 1
        )

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        @jit
        def sgd_step(
            states: TrainingStateSeparateNetworks, sample: reverb.ReplaySample
        ) -> Tuple[TrainingStateSeparateNetworks, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

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
                """Gets values of the joint oberservation space"""
                o = jax.tree_map(
                    lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])), observation
                )
                behavior_values = networks[net_key].critic_network.apply(
                    states.critic_params[net_key], o
                )

                behavior_values = jnp.reshape(behavior_values, reward.shape[0:2])
                return behavior_values

            # TODO (Ruan): Double check this
            agent_nets = trainer.store.trainer_agent_net_keys
            index = "agent_0"

            # create joint obervation space
            # agent_keys = []
            all_obs_list = []
            for key in agent_nets.keys():
                # agent_keys.append(key)
                all_obs_list.append(observations[key].observation)
            joint_obs = jax.numpy.concatenate(all_obs_list, axis=-1)

            behavior_values = {}
            behavior_value = get_behavior_values(
                agent_nets[index], rewards[index], joint_obs
            )

            # TODO:
            # Create duplicate values to generate multiple GAE
            for key in agent_nets.keys():
                behavior_values[key] = behavior_value

            # Vmap over batch dimension
            batch_gae_advantages = jax.vmap(trainer.store.gae_fn, in_axes=0)

            advantages = {}
            target_values = {}
            # Assume rewards are the same for cooperative setting. Check this.
            advantage, target_value = batch_gae_advantages(
                rewards[index], discounts[index], behavior_values[index]
            )

            for key in rewards.keys():
                advantages[key] = advantage
                target_values[key] = target_value

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_sequences, num_steps, ..]
            observations, actions, behavior_log_probs, behavior_values = jax.tree_map(
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
            assert batch_size % trainer.store.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.num_minibatches)
            batch = jax.tree_map(
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
                length=trainer.store.num_epochs,
            )

            # Set the metrics
            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_policy_params"] = optax.global_norm(states.policy_params)
            metrics["norm_critic_params"] = optax.global_norm(states.critic_params)
            metrics["observations_mean"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(
                        lambda x: jnp.abs(jnp.mean(x, axis=(0, 1))), observations
                    ),
                    num_batch_dims=0,
                )
            )
            metrics["observations_std"] = jnp.mean(
                utils.batch_concat(
                    jax.tree_map(lambda x: jnp.std(x, axis=(0, 1)), observations),
                    num_batch_dims=0,
                )
            )
            metrics["rewards_mean"] = jax.tree_map(
                lambda x: jnp.mean(jnp.abs(jnp.mean(x, axis=(0, 1)))), rewards
            )
            metrics["rewards_std"] = jax.tree_map(
                lambda x: jnp.std(x, axis=(0, 1)), rewards
            )

            new_states = TrainingStateSeparateNetworks(
                policy_params=new_policy_params,
                critic_params=new_critic_params,
                policy_opt_states=new_policy_opt_states,
                critic_opt_states=new_critic_opt_states,
                random_key=new_key,
            )
            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:
            """_summary_"""
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
            random_key, _ = jax.random.split(trainer.store.key)

            states = TrainingStateSeparateNetworks(
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
            trainer.store.key = new_states.random_key

            # TODO (Ruan): this could probably be done in one loop
            # instead of a different policy and critic loop to save time
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
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPPOStepConfig
