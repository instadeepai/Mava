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

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree
from acme.jax import utils
from jax import jit

from mava.components.jax import Component
from mava.components.jax.training import Batch, Step, TrainingState
from mava.components.jax.training.base import BatchDQN
from mava.core_jax import SystemTrainer


@dataclass
class MADQNStepConfig:
    target_update_period: int = 10
    # discount: float = 0.99 this is defined somewhere else, I guess in transition.


class MADQNStep(Step):
    def __init__(
            self,
            config: MADQNStepConfig = MADQNStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        # @jit
        def sgd_step(
                states: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

            # Extract the data.
            data = sample.data

            observations, new_observations, actions, rewards, discounts, extra = (
                data.observations,
                data.next_observations,
                data.actions,
                data.rewards,
                data.discounts,
                data.extras,
            )

            discounts = tree.map_structure(
                lambda x: x * self.config.discount,
                discounts
                # lambda x: x * 0.99, discounts  # TODO: this is a hack, fix it
            )

            trajectories = BatchDQN(
                observations=observations,
                next_observations=new_observations,
                actions=actions,
                rewards=rewards,
                discounts=discounts,
            )

            batch = trajectories

            # Calling epoch_update_fn on the batch data to update network parameters.
            with jax.disable_jit():
                (new_key, new_params, new_target_params, new_opt_states, _,), metrics \
                    = jax.lax.scan(
                    trainer.store.epoch_update_fn,
                    (states.random_key, states.params, states.target_params,
                     states.opt_states, batch), {},
                    length=trainer.store.num_epochs,
                )
            # removed the lax.scan in the following

            # Periodically update the target network to the updated network
            new_target_params = rlax.periodic_update(
                new_params,
                new_target_params,
                trainer.store.training_steps,
                self.config.target_update_period,
            )

            # Update the training states.
            new_states = TrainingState(
                params=new_params,
                target_params=new_target_params,
                opt_states=new_opt_states,
                random_key=new_key,
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
            metrics["rewards_mean"] = jnp.mean(
                utils.batch_concat(rewards, num_batch_dims=0)
            )
            metrics["rewards_std"] = jnp.std(
                utils.batch_concat(rewards, num_batch_dims=0)
            )

            return new_states, metrics

        def step(sample: reverb.ReplaySample) -> Tuple[Dict[str, jnp.ndarray]]:

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            networks = trainer.store.networks["networks"]
            target_networks = trainer.store.target_networks["networks"]
            params = {net_key: networks[net_key].params for net_key in networks.keys()}
            target_params = {
                net_key: target_networks[net_key].params
                for net_key in target_networks.keys()
            }
            opt_states = trainer.store.opt_states
            random_key, _ = jax.random.split(trainer.store.key)

            states = TrainingState(
                params=params,
                target_params=target_params,
                opt_states=opt_states,
                random_key=random_key,
            )

            new_states, metrics = sgd_step(states, sample)

            # Set the new variables
            # TODO (dries): key is probably not being store correctly.
            # The variable client might lose reference to it when checkpointing.
            # We also need to add the optimizer and random_key to the variable
            # server.
            trainer.store.key = new_states.random_key

            networks = trainer.store.networks["networks"]
            target_networks = trainer.store.target_networks["networks"]

            params = {net_key: networks[net_key].params for net_key in networks.keys()}

            # Updating the networks:
            for net_key in params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.networks["networks"][net_key].params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.params[net_key][param_key]

            # Update the optimizer
            for net_key in params.keys():
                # This needs to be in the loop to not lose the reference.
                trainer.store.opt_states[net_key] = new_states.opt_states[net_key]

            # Update the target networks
            target_params = {
                net_key: target_networks[net_key].params
                for net_key in target_networks.keys()
            }

            for net_key in target_params.keys():
                # This below forloop is needed to not lose the param reference.
                net_params = trainer.store.target_networks["networks"][net_key].params
                for param_key in net_params.keys():
                    net_params[param_key] = new_states.target_params[net_key][param_key]

            # Set the metrics
            trainer.store.metrics = metrics

            return metrics

        trainer.store.step_fn = step

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "madqn_step_fn"

    @staticmethod
    def config_class() -> Callable:
        """_summary_"""
        return MADQNStepConfig
