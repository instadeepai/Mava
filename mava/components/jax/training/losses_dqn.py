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

"""Trainer components for calculating losses."""

import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import rlax

from mava.components.jax.training.base import Loss
from mava.core_jax import SystemTrainer


@dataclass
class MADQNLossConfig:
    # clipping_epsilon: float = 0.2
    # clip_value: bool = True
    # entropy_cost: float = 0.01
    # value_cost: float = 0.5
    max_abs_reward: float = 1.0
    gamma: float = 0.99


class MADQNLoss(Loss):
    """Deep q learning.

    This matches the original DQN loss: https://arxiv.org/abs/1312.5602.
    It differs by two aspects that improve it on the optimization side
    - it uses a square loss instead of the Huber one.
    """

    def __init__(
        self,
        config: MADQNLossConfig = MADQNLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Creates the grad function of the loss and adds it to the trainer.store."""

        @chex.assert_max_traces(n=1)
        def loss_grad_fn(
            trainer_network: Any,
            params: Any,
            target_params: Any,
            observations: Any,
            next_observations: Any,
            actions: Dict[str, jnp.ndarray],
            discounts: Any,
            rewards: Any,
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Gradient of loss.

            Args:
                params: parameters of all models.
                target_params: parameters of all target model.
                observations: observations of all the agents.
                next_observations: next observations of all the agents.
                actions: actions of all the agents.
                discounts: discounts of all the agents.
                rewards: rewards of all the agents.

            Returns:
                grads: gradients of loss with respect to all the parameters.
                extra: extra information.
            """

            grads = {}
            loss = {}
            loss_info = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer_network[agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def loss_fn(
                    params: Any,
                    target_params: Any,
                    observations: Any,
                    next_observations: Any,
                    actions: jnp.ndarray,
                    discount: Any,
                    rewards: Any,
                    next_legal_actions: Any,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    # Forward pass.
                    q_tm1 = network.forward_fn(params, observations)
                    q_t_value = network.forward_fn(target_params, next_observations)
                    q_t_selector = network.forward_fn(params, next_observations)

                    q_t_selector = jnp.where(
                        next_legal_actions.astype(bool),
                        q_t_selector,
                        jnp.finfo(q_t_selector.dtype).min,
                    )

                    d_t = (discount * self.config.gamma).astype(jnp.float32)
                    # Cast and clip rewards.
                    r_t = jnp.clip(
                        rewards,
                        -self.config.max_abs_reward,
                        self.config.max_abs_reward,
                    ).astype(jnp.float32)

                    # Compute Q-learning TD-error.
                    batch_error = jax.vmap(rlax.double_q_learning)
                    td_error = batch_error(
                        q_tm1, actions, r_t, d_t, q_t_value, q_t_selector
                    )
                    loss = jnp.mean(rlax.l2_loss(td_error))
                    loss_info = {"loss_total": loss}
                    return loss, loss_info

                (loss[agent_key], loss_info[agent_key]), grads[
                    agent_key
                ] = jax.value_and_grad(loss_fn, has_aux=True)(
                    params[agent_net_key],
                    target_params[agent_net_key],
                    observations[agent_key].observation,
                    next_observations[agent_key].observation,
                    actions[agent_key],
                    discounts[agent_key],
                    rewards[agent_key],
                    next_observations[agent_key].legal_actions,
                )
                loss_info["total_loss"] = loss[agent_key]

            return grads, loss_info

        # Save the gradient function.
        trainer.store.grad_fn = jax.jit(
            functools.partial(loss_grad_fn, trainer.store.networks["networks"])
        )

    @staticmethod
    def config_class() -> Callable:
        """Returns the config class for this loss."""
        return MADQNLossConfig

    @staticmethod
    def name() -> str:
        """Returns name of the component."""
        return "loss"
