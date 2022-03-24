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

"""Trainer components for system updating"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from mava.components.jax.training import Batch, Utility
from mava.core_jax import SystemTrainer


@dataclass
class MAPGMinibatchUpdateConfig:
    learning_rate: float = 1e-3
    adam_epsilon: float = 1e-5
    max_gradient_norm: float = 0.5


class MAPGMinibatchUpdate(Utility):
    def __init__(
        self,
        config: MAPGMinibatchUpdateConfig = MAPGMinibatchUpdateConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""
        grad_fn = trainer.attr.grad_fn
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_gradient_norm),
            optax.scale_by_adam(eps=self.config.adam_epsilon),
            optax.scale(-self.config.learning_rate),
        )

        def model_update_minibatch(
            carry: Tuple[Any, optax.OptState], minibatch: Batch
        ) -> Tuple[Tuple[Any, optax.OptState], Dict[str, jnp.ndarray]]:
            """Performs model update for a single minibatch."""
            params, opt_state = carry
            # Normalize advantages at the minibatch level before using them.
            advantages = (
                minibatch.advantages - jnp.mean(minibatch.advantages, axis=0)
            ) / (jnp.std(minibatch.advantages, axis=0) + 1e-8)
            gradients, metrics = grad_fn(
                params,
                minibatch.observations,
                minibatch.actions,
                minibatch.behavior_log_probs,
                minibatch.target_values,
                advantages,
                minibatch.behavior_values,
            )

            # Apply updates
            updates, opt_state = optimizer.update(gradients, opt_state)
            params = optax.apply_updates(params, updates)

            metrics["norm_grad"] = optax.global_norm(gradients)
            metrics["norm_updates"] = optax.global_norm(updates)
            return (params, opt_state), metrics

        trainer.attr.minibatch_update_fn = model_update_minibatch

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "minibatch_update_fn"


@dataclass
class MAPGEpochUpdateConfig:
    num_epochs: int = 4
    num_minibatches: int = 1
    batch_size: int = 256


class MAPGEpochUpdate(Utility):
    def __init__(
        self,
        config: MAPGEpochUpdateConfig = MAPGEpochUpdateConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def model_update_epoch(
            carry: Tuple[jnp.ndarray, Any, optax.OptState, Batch],
            unused_t: Tuple[()],
        ) -> Tuple[
            Tuple[jnp.ndarray, Any, optax.OptState, Batch],
            Dict[str, jnp.ndarray],
        ]:
            """Performs model updates based on one epoch of data."""
            key, params, opt_state, batch = carry
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, self.config.batch_size)
            shuffled_batch = jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_map(
                lambda x: jnp.reshape(
                    x, [self.config.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            (params, opt_state), metrics = jax.lax.scan(
                trainer.attr.model_update_minibatch,
                (params, opt_state),
                minibatches,
                length=self.config.num_minibatches,
            )

            return (key, params, opt_state, batch), metrics

        trainer.attr.epoch_update_fn = model_update_epoch

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "minibatch_update_fn"
