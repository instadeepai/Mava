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

"""Trainer components for system updating."""

import abc
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import optax
from acme.jax import networks as networks_lib
from jax.random import KeyArray

from mava import constants
from mava.callbacks import Callback
from mava.components.building.optimisers import Optimisers
from mava.components.training.base import Batch, Utility
from mava.components.training.losses import Loss
from mava.components.training.step import Step
from mava.components.training.trainer import BaseTrainerInit
from mava.core_jax import SystemTrainer
from mava.utils.jax_training_utils import (
    compute_running_mean_var_count,
    construct_norm_axes_list,
    update_and_normalize_observations,
)


class MinibatchUpdate(Utility):
    @abc.abstractmethod
    def __init__(self, config: Any) -> None:
        """Abstract component defining a mini-batch update."""
        self.config = config

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "minibatch_update"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseTrainerInit required to set up trainer.store.trainer_agents and
        trainer.store.trainer_agent_net_keys
        Optmisers required to set up trainer.store.policy_optimiser and
        trainer.store.critic_optimiser.
        Loss required to set up trainer.store.policy_grad_fn
        and trainer.store.critic_grad_fn.

        Returns:
            List of required component classes.
        """
        return [BaseTrainerInit, Loss, Optimisers]


@dataclass
class MAPGMinibatchUpdateConfig:
    normalize_advantage: bool = True
    normalize_target_values: bool = False
    normalize_observations: bool = False
    normalize_axes: Union[List[Any], None] = field(
        default_factory=lambda: None
    )  # [1, 2, (4,7), [10,12]]


class MAPGMinibatchUpdate(MinibatchUpdate):
    def __init__(
        self,
        config: MAPGMinibatchUpdateConfig = MAPGMinibatchUpdateConfig(),
    ):
        """Component defines a multi-agent policy gradient mini-batch update.

        Args:
            config: MAPGMinibatchUpdateConfig.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Create and store MAPG mini-batch update function.

        Creates a default critic and policy optimisers if none
        are provided in the config.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        # Initilaise target values running mean/std function here
        if self.config.normalize_target_values:
            trainer.store.target_running_stats_fn = compute_running_mean_var_count

        # Initilaise observations running mean/std function here
        if self.config.normalize_observations:
            observation_stats = trainer.store.norm_params[
                constants.OBS_NORM_STATE_DICT_KEY
            ]
            obs_shape = observation_stats[list(observation_stats.keys())[0]][
                "mean"
            ].shape
            norm_axes = construct_norm_axes_list(
                trainer.store.obs_normalisation_start,
                self.config.normalize_axes,
                obs_shape,
            )
            trainer.store.norm_obs_running_stats_fn = partial(
                update_and_normalize_observations,
                axes=norm_axes,
            )

        def model_update_minibatch(
            carry: Tuple[
                networks_lib.Params, networks_lib.Params, optax.OptState, optax.OptState
            ],
            minibatch: Batch,
        ) -> Tuple[Tuple[Any, Any, optax.OptState, optax.OptState], Dict[str, Any]]:
            """Performs model update for a single minibatch."""
            policy_params, critic_params, policy_opt_states, critic_opt_states = carry

            # Flatten if it is recurrent
            if list(minibatch.policy_states.values())[0]:
                agent_0_t_vals = list(minibatch.target_values.values())[0]
                num_sequences = agent_0_t_vals.shape[0]
                num_steps = agent_0_t_vals.shape[1]
                batch_size = num_sequences * num_steps

                minibatch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), minibatch
                )
            else:
                raise ValueError("Not testing feedforward at the moment!")

            # Normalize advantages at the minibatch level before using them.
            if self.config.normalize_advantage:
                advantages = jax.tree_util.tree_map(
                    lambda x: (x - jnp.mean(x, axis=0)) / (jnp.std(x, axis=0) + 1e-8),
                    minibatch.advantages,
                )
            else:
                advantages = minibatch.advantages

            # jax.debug.print("🤯 Actions mask: {x} 🤯", x=minibatch.actions["agent_0"].reshape((5, 19)))
            # jax.debug.print("🤯 Loss mask: {x} 🤯", x=minibatch.loss_masks["agent_0"].reshape((5, 19)))

            # Calculate the gradients and agent metrics.
            policy_gradients, policy_agent_metrics = trainer.store.policy_grad_fn(
                policy_params,
                minibatch.policy_states,
                minibatch.observations,
                minibatch.loss_masks,
                minibatch.actions,
                minibatch.behavior_log_probs,
                advantages,
            )

            # Calculate the gradients and agent metrics.
            critic_gradients, critic_agent_metrics = trainer.store.critic_grad_fn(
                critic_params,
                minibatch.observations,
                minibatch.loss_masks,
                minibatch.target_values,
                minibatch.behavior_values,
            )

            metrics = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                # Update the policy networks and optimisers.
                # Apply updates
                # TODO (dries): Use one optimiser per network type here and not
                # just one.
                (
                    policy_updates,
                    policy_opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
                ) = trainer.store.policy_optimiser.update(
                    policy_gradients[agent_key],
                    policy_opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
                )
                policy_params[agent_net_key] = optax.apply_updates(
                    policy_params[agent_net_key], policy_updates
                )

                policy_agent_metrics[agent_key]["norm_policy_grad"] = optax.global_norm(
                    policy_gradients[agent_key]
                )
                policy_agent_metrics[agent_key][
                    "norm_policy_updates"
                ] = optax.global_norm(policy_updates)
                metrics[agent_key] = policy_agent_metrics[agent_key]

                # Update the critic networks and optimisers.
                # Apply updates
                # TODO (dries): Use one optimiser per network type here and not
                # just one.
                (
                    critic_updates,
                    critic_opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
                ) = trainer.store.critic_optimiser.update(
                    critic_gradients[agent_key],
                    critic_opt_states[agent_net_key][constants.OPT_STATE_DICT_KEY],
                )
                critic_params[agent_net_key] = optax.apply_updates(
                    critic_params[agent_net_key], critic_updates
                )

                critic_agent_metrics[agent_key]["norm_critic_grad"] = optax.global_norm(
                    critic_gradients[agent_key]
                )
                critic_agent_metrics[agent_key][
                    "norm_critic_updates"
                ] = optax.global_norm(critic_updates)
                # TODO (Ruan): double check that this was done correctly
                metrics[agent_key].update(critic_agent_metrics[agent_key])

            return (
                policy_params,
                critic_params,
                policy_opt_states,
                critic_opt_states,
            ), metrics

        trainer.store.minibatch_update_fn = model_update_minibatch


class EpochUpdate(Utility):
    @abc.abstractmethod
    def __init__(self, config: Any) -> None:
        """Abstract component for performing model updates from an entire epoch."""
        self.config = config

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "epoch_update"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        MinibatchUpdate required to set up trainer.store.minibatch_update_fn.

        Returns:
            List of required component classes.
        """
        return [Step, MinibatchUpdate]


@dataclass
class MAPGEpochUpdateConfig:
    num_epochs: int = 4
    num_minibatches: int = 1


class MAPGEpochUpdate(EpochUpdate):
    def __init__(
        self,
        config: MAPGEpochUpdateConfig = MAPGEpochUpdateConfig(),
    ):
        """Component defines a multi-agent policy gradient epoch-level update.

        Args:
            config: MAPGEpochUpdateConfig.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Define and store the epoch update function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """
        trainer.store.num_epochs = self.config.num_epochs
        trainer.store.num_minibatches = self.config.num_minibatches

        def model_update_epoch(
            carry: Tuple[KeyArray, Any, Any, optax.OptState, optax.OptState, Batch],
            unused_t: Tuple[()],
        ) -> Tuple[
            Tuple[KeyArray, Any, Any, optax.OptState, optax.OptState, Batch],
            Dict[str, jnp.ndarray],
        ]:
            """Performs model updates based on one epoch of data."""
            (
                key,
                policy_params,
                critic_params,
                policy_opt_states,
                critic_opt_states,
                batch,
            ) = carry

            base_key, shuffle_key = jax.random.split(key)

            # Note (dries): This computation is only performed once at compilation tome
            # and not at every epoch step.
            batch_shape = jax.tree_util.tree_leaves(
                list(batch.observations.values())[0].observation
            )[0].shape[0]

            permutation = jax.random.permutation(
                shuffle_key, batch_shape
            )

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [self.config.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            (
                new_policy_params,
                new_critic_params,
                new_policy_opt_states,
                new_critic_opt_states,
            ), metrics = jax.lax.scan(
                trainer.store.minibatch_update_fn,
                (policy_params, critic_params, policy_opt_states, critic_opt_states),
                minibatches,
                length=self.config.num_minibatches,
            )

            return (
                base_key,
                new_policy_params,
                new_critic_params,
                new_policy_opt_states,
                new_critic_opt_states,
                batch,
            ), metrics

        trainer.store.epoch_update_fn = model_update_epoch
