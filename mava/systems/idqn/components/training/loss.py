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
from typing import Dict, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import rlax
from acme.agents.jax.dqn.learning_lib import ReverbUpdate

from mava.components.training.losses import Loss
from mava.core_jax import SystemTrainer
from mava.types import OLT


@dataclass
class IDQNLossConfig:
    gamma: float = 0.99
    importance_sampling_exponent: float = 0.6


class IDQNLoss(Loss):
    def __init__(
        self,
        config: IDQNLossConfig = IDQNLossConfig(),
    ):
        """Component defines a MAPGWithTrustRegionClipping loss function.

        Args:
            config: MAPGTrustRegionClippingLossConfig.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Create and store IDQN loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def policy_loss_grad_fn(
            policy_params: Dict[str, hk.Params],
            target_policy_params: Dict[str, hk.Params],
            observations: Dict[str, OLT],
            actions: Dict[str, chex.Array],
            rewards: Dict[str, chex.Array],
            next_observations: Dict[str, OLT],
            discounts: Dict[str, chex.Array],
            probs: chex.Array,
            keys: chex.Array,
        ) -> Tuple[Dict[str, chex.Array], Dict[str, chex.Array], Dict[str, chex.Array]]:
            """Surrogate loss using clipped probability ratios.

            Args:
                policy_params: policy network parameters.
                target_policy_params: target network parameters
                observations: agent observations at timestep t.
                actions: actions the agents took.
                rewards: rewards given to the agent
                next_observations: agent observations at timestep t+1
                discounts: terminal agent mask (dm_env discounts)
                probs: probabilities for priotised experience replay
                keys: keys of reverb table entries

            Returns:
                Tuple[policy gradients, policy loss information]
            """

            grads = {}
            loss_info = {}
            reverb_updates = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks[agent_net_key]

                def policy_loss_fn(
                    policy_params: hk.Params,
                    target_policy_params: hk.Params,
                    observations: chex.Array,
                    actions: chex.Array,
                    rewards: chex.Array,
                    next_observations: chex.Array,
                    discounts: chex.Array,
                    masks: chex.Array,
                ) -> Tuple[chex.Array, Tuple[Dict[str, chex.Array], ReverbUpdate]]:
                    """Inner policy loss function: see outer function for parameters."""

                    # Feedforward actor.
                    q_tm1 = network.forward(policy_params, observations)
                    q_t_value = network.forward(target_policy_params, next_observations)
                    q_t_selector = network.forward(policy_params, next_observations)

                    q_t_selector = jnp.where(masks == 1.0, q_t_selector, -99999)  # TODO

                    batch_double_q_learning_loss_fn = jax.vmap(
                        rlax.double_q_learning, (0, 0, 0, 0, 0, 0, None)
                    )

                    error = batch_double_q_learning_loss_fn(
                        q_tm1,
                        actions,
                        rewards,
                        discounts * self.config.gamma,
                        q_t_value,
                        q_t_selector,
                        True,
                    )

                    batch_loss = rlax.l2_loss(error)

                    importance_weights = (1.0 / probs).astype(jnp.float32)
                    importance_weights **= self.config.importance_sampling_exponent
                    importance_weights /= jnp.max(importance_weights)

                    # Weigthing loss by probability transition was chosen
                    loss = jnp.mean(importance_weights * batch_loss)
                    # makes sure prio never exceeds one
                    reverb_update = ReverbUpdate(keys=keys, priorities=jnp.abs(error))
                    loss_info = (
                        {"policy_loss_total": loss},
                        reverb_update,
                    )

                    return loss, loss_info

                grad_fn = jax.grad(policy_loss_fn, has_aux=True)
                (
                    grads[agent_key],
                    (loss_info[agent_key], reverb_updates[agent_key]),
                ) = grad_fn(
                    policy_params[agent_net_key],
                    target_policy_params[agent_net_key],
                    observations[agent_key].observation,
                    actions[agent_key],
                    rewards[agent_key],
                    next_observations[agent_key].observation,
                    discounts[agent_key],
                    next_observations[agent_key].legal_actions,
                )
            return grads, loss_info, reverb_updates

        # Save the gradient funcitons.
        trainer.store.policy_grad_fn = policy_loss_grad_fn
