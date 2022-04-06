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

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import reverb
from acme.jax import utils

from mava.components.jax import Component
from mava.components.jax.training import Batch, TrainingState, Utility
from mava.core_jax import SystemTrainer


@dataclass
class DefaultStepConfig:
    random_key: jnp.ndarray = 42


class DefaultStep(Component):
    def __init__(
        self,
        config: DefaultStepConfig = DefaultStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_init(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        # Initialise training state (parameters and optimiser state).
        trainer.store.state = trainer.store.make_initial_state(self.config.random_key)

    def on_training_step(self, trainer: SystemTrainer) -> None:
        """Does a step of SGD and logs the results."""

        # Do a batch of SGD.
        sample = next(trainer.store.iterator)
        self._state, results = trainer.store.sgd_step(trainer.store.state, sample)

        # Update our counts and record it.
        # counts = self._counter.increment(steps=1) # TODO: add back in later

        # Snapshot and attempt to write logs.
        # self._logger.write({**results, **counts})
        trainer.store.logger.write({**results})

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "step"


@dataclass
class MAPGWithTrustRegionStepConfig:
    discount: float = 0.99


class MAPGWithTrustRegionStep(Utility):
    def __init__(
        self,
        config: MAPGWithTrustRegionStepConfig = MAPGWithTrustRegionStepConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def sgd_step(
            state: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

            # Extract the data.
            data = sample.data
            observations, actions, rewards, termination, extra = (
                data.observation,
                data.action,
                data.reward,
                data.discount,
                data.extras,
            )
            discounts = termination * self.config.discount
            behavior_log_probs = extra["log_prob"]

            def get_behavior_values(params: Any, observations: Any) -> jnp.ndarray:
                o = jax.tree_map(
                    lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])), observations
                )
                _, behavior_values = trainer.store.networks.network.apply(params, o)
                behavior_values = jnp.reshape(behavior_values, rewards.shape[0:2])
                return behavior_values

            behavior_values = get_behavior_values(state.params, observations)

            # Vmap over batch dimension
            batch_gae_advantages = jax.vmap(trainer.store.gae_fn, in_axes=0)
            advantages, target_values = batch_gae_advantages(
                rewards, discounts, behavior_values
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
            assert len(target_values.shape) > 1
            num_sequences = target_values.shape[0]
            num_steps = target_values.shape[1]
            batch_size = num_sequences * num_steps
            assert batch_size % trainer.store.num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, trainer.store.num_minibatches)
            batch = jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            # Compute gradients.
            trainer.store.grad_fn = jax.grad(trainer.store.loss_fn, has_aux=True)

            params = state.params
            opt_state = state.opt_state
            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            (key, params, opt_state, _), metrics = jax.lax.scan(
                trainer.store.epoch_update_fn,
                (state.random_key, params, opt_state, batch),
                (),
                length=trainer.store.num_epochs,
            )

            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["norm_params"] = optax.global_norm(params)
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
            metrics["rewards_mean"] = jnp.mean(jnp.abs(jnp.mean(rewards, axis=(0, 1))))
            metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))
            new_state = TrainingState(
                params=params, opt_state=opt_state, random_key=key
            )
            return new_state, metrics

        trainer.store.step_fn = sgd_step

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "step_fn"
