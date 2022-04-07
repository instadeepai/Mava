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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from acme.jax import networks as networks_lib
from jax.random import KeyArray
from optax._src import base as optax_base

from mava.components.jax.training import Batch, Utility
from mava.core_jax import SystemTrainer


@dataclass
class MAPGMinibatchUpdateConfig:
    learning_rate: float = 1e-3
    adam_epsilon: float = 1e-5
    max_gradient_norm: float = 0.5
    optimizer: Optional[optax_base.GradientTransformation] = (None,)


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

        if not self.config.optimizer:
            trainer.store.optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.max_gradient_norm),
                optax.scale_by_adam(eps=self.config.adam_epsilon),
                optax.scale(-self.config.learning_rate),
            )
        else:
            trainer.store.optimizer = self.config.optimizer

        # Initialize optimizers.

        # TODO (dries): Implement for multiple policy and critic networks.
        assert len(trainer.store.networks["networks"]) == 1

        network = list(trainer.store.networks["networks"].values())[0]

        trainer.store.opt_state = trainer.store.optimizer.init(
            network.params
        )  # pytype: disable=attribute-error

        def model_update_minibatch(
            carry: Tuple[networks_lib.Params, optax.OptState], minibatch: Batch
        ) -> Tuple[Tuple[Any, optax.OptState], Dict[str, jnp.ndarray]]:
            """Performs model update for a single minibatch."""
            params, opt_state = carry
            # Normalize advantages at the minibatch level before using them.
            advantages = (
                minibatch.advantages - jnp.mean(minibatch.advantages, axis=0)
            ) / (jnp.std(minibatch.advantages, axis=0) + 1e-8)

            # TODO (dries): Implement this for the multiagent case.
            net_key = list(trainer.store.networks["networks"].keys())[0]

            gradients, metrics = trainer.store.grad_fn(
                params,
                minibatch.observations,
                minibatch.actions,
                minibatch.behavior_log_probs,
                minibatch.target_values,
                advantages,
                minibatch.behavior_values,
            )

            # Apply updates
            updates, trainer.store.opt_state = trainer.store.optimizer.update(
                gradients, opt_state
            )
            trainer.store.networks["networks"][net_key].params = optax.apply_updates(
                params, updates
            )

            metrics["norm_grad"] = optax.global_norm(gradients)
            metrics["norm_updates"] = optax.global_norm(updates)
            return (params, opt_state), metrics

        trainer.store.minibatch_update_fn = model_update_minibatch

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
        trainer.store.num_epochs = self.config.num_epochs
        trainer.store.num_minibatches = self.config.num_minibatches

        def model_update_epoch(
            carry: Tuple[KeyArray, Any, optax.OptState, Batch],
            unused_t: Tuple[()],
        ) -> Tuple[
            Tuple[KeyArray, Any, optax.OptState, Batch],
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
                trainer.store.minibatch_update_fn,
                (params, opt_state),
                minibatches,
                length=self.config.num_minibatches,
            )

            return (key, params, opt_state, batch), metrics

        trainer.store.epoch_update_fn = model_update_epoch

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "epoch_update_fn"
