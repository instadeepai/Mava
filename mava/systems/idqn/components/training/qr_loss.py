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
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import rlax

from mava.core_jax import SystemTrainer
from mava.systems.idqn.components.training.loss import IDQNLoss, IDQNLossConfig


@dataclass
class QrIDQNLossConfig(IDQNLossConfig):
    huber_param: float = 1.0


class QrIDQNLoss(IDQNLoss):
    def __init__(self, config: QrIDQNLossConfig = QrIDQNLossConfig()) -> None:
        """Initialize."""
        super().__init__(config)

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Create and store Quantile regression IDQN loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def policy_loss_grad_fn(
            policy_params: Any,
            target_policy_params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            rewards: Dict[str, jnp.ndarray],
            next_observations: Any,
            discounts: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """QR-DQN Loss.

            Args:
                policy_params: policy network parameters.
                target_policy_params: target network parameters
                observations: agent observations at timestep t.
                actions: actions the agents took.
                rewards: rewards given to the agent
                next_observations: agent observations at timestep t+1
                discounts: terminal agent mask (dm_env discounts)

            Returns:
                Tuple[policy gradients, policy loss information]
            """

            policy_grads = {}
            loss_info_policy = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks[agent_net_key]

                def policy_loss_fn(
                    policy_params: Any,
                    target_policy_params: Any,
                    observations: Any,
                    actions: jnp.ndarray,
                    rewards: jnp.ndarray,
                    next_observations: Any,
                    discounts: jnp.ndarray,
                    masks: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    """Inner policy loss function: see outer function for parameters."""

                    _, dist_q_tm1 = network.forward(policy_params, observations)
                    _, dist_q_target_t = network.forward(
                        target_policy_params, next_observations
                    )
                    _, q_t_selector_dist = network.forward(
                        policy_params, next_observations
                    )

                    cond = jnp.expand_dims(masks == 1.0, -1)
                    q_t_selector_dist = jnp.where(cond, q_t_selector_dist, -jnp.inf)

                    # Swap distribution and action dimension, since
                    # rlax.quantile_q_learning expects it that way.
                    dist_q_tm1 = jnp.swapaxes(dist_q_tm1, 1, 2)
                    dist_q_target_t = jnp.swapaxes(dist_q_target_t, 1, 2)

                    num_atoms = dist_q_tm1.shape[1]
                    quantiles = (jnp.arange(num_atoms, dtype=float) + 0.5) / num_atoms
                    batch_quantile_q_learning = jax.vmap(
                        rlax.quantile_q_learning, in_axes=(0, None, 0, 0, 0, 0, 0, None)
                    )
                    losses = batch_quantile_q_learning(
                        dist_q_tm1,
                        quantiles,
                        actions,
                        rewards,
                        discounts * self.config.gamma,
                        q_t_selector_dist,
                        dist_q_target_t,
                        self.config.huber_param,
                    )
                    loss = jnp.mean(losses)
                    extra = {"policy_loss_total": loss}
                    return loss, extra

                policy_grads[agent_key], loss_info_policy[agent_key] = jax.grad(
                    policy_loss_fn, has_aux=True
                )(
                    policy_params[agent_net_key],
                    target_policy_params[agent_net_key],
                    observations[agent_key].observation,
                    actions[agent_key],
                    rewards[agent_key],
                    next_observations[agent_key].observation,
                    discounts[agent_key],
                    next_observations[agent_key].legal_actions,
                )
            return policy_grads, loss_info_policy

        # Save the gradient funcitons.
        trainer.store.policy_grad_fn = policy_loss_grad_fn
