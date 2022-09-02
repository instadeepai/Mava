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
import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import rlax

from mava.callbacks import Callback
from mava.components.jax import Component, training
from mava.core_jax import SystemTrainer


class Loss(Component):
    @abc.abstractmethod
    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """[summary]"""

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "loss"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseTrainerInit required to set up trainer.store.trainer_agents,
        trainer.store.trainer_agent_net_keys and trainer.store.networks.

        Returns:
            List of required component classes.
        """
        return [
            training.BaseTrainerInit
        ]  # import from training to avoid partial dependency


@dataclass
class MAPGTrustRegionClippingLossConfig:
    clipping_epsilon: float = 0.2
    clip_value: bool = True
    entropy_cost: float = 0.01
    value_cost: float = 0.5


class MAPGWithTrustRegionClippingLoss(Loss):
    def __init__(
        self,
        config: MAPGTrustRegionClippingLossConfig = MAPGTrustRegionClippingLossConfig(),
    ):
        """Component defines a MAPGWithTrustRegionClipping loss function.

        Args:
            config: MAPGTrustRegionClippingLossConfig.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Create and store MAPGWithTrustRegionClipping loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def loss_grad_fn(
            params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
            behavior_values: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios.

            Args:
                params: network parameters.
                observations: agent observations.
                actions: actions the agents took.
                behaviour_log_probs: log probablity of action taken.
                target_values: values computed using target networks.
                advantages: advantage estimation values per agent.
                behavior_values: estimated value from the critic.

            Returns:
                Tuple[gradients, loss info]
            """

            grads = {}
            loss_info = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def loss_fn(
                    params: Any,
                    observations: Any,
                    actions: jnp.ndarray,
                    behaviour_log_probs: jnp.ndarray,
                    target_values: jnp.ndarray,
                    advantages: jnp.ndarray,
                    behavior_values: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    """Inner loss function: see outer function for parameters."""
                    distribution_params, values = network.network.apply(
                        params, observations
                    )
                    log_probs = network.log_prob(distribution_params, actions)
                    entropy = network.entropy(distribution_params)
                    # Compute importance sampling weights:
                    # current policy / behavior policy.
                    rhos = jnp.exp(log_probs - behaviour_log_probs)
                    clipping_epsilon = self.config.clipping_epsilon

                    policy_loss = rlax.clipped_surrogate_pg_loss(
                        rhos, advantages, clipping_epsilon
                    )

                    # Value function loss. Exclude the bootstrap value
                    unclipped_value_error = target_values - values
                    unclipped_value_loss = unclipped_value_error**2

                    if self.config.clip_value:
                        # Clip values to reduce variablility during critic training.
                        clipped_values = behavior_values + jnp.clip(
                            values - behavior_values,
                            -clipping_epsilon,
                            clipping_epsilon,
                        )
                        clipped_value_error = target_values - clipped_values
                        clipped_value_loss = clipped_value_error**2
                        value_loss = jnp.mean(
                            jnp.fmax(unclipped_value_loss, clipped_value_loss)
                        )
                    else:
                        value_loss = jnp.mean(unclipped_value_loss)

                    # Entropy regulariser.
                    entropy_loss = -jnp.mean(entropy)

                    total_loss = (
                        policy_loss
                        + value_loss * self.config.value_cost
                        + entropy_loss * self.config.entropy_cost
                    )

                    loss_info = {
                        "loss_total": total_loss,
                        "loss_policy": policy_loss,
                        "loss_value": value_loss,
                        "loss_entropy": entropy_loss,
                    }

                    return total_loss, loss_info

                grads[agent_key], loss_info[agent_key] = jax.grad(
                    loss_fn, has_aux=True
                )(
                    params[agent_net_key],
                    observations[agent_key].observation,
                    actions[agent_key],
                    behaviour_log_probs[agent_key],
                    target_values[agent_key],
                    advantages[agent_key],
                    behavior_values[agent_key],
                )
            return grads, loss_info

        # Save the gradient funciton.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGTrustRegionClippingLossConfig


@dataclass
class MAPGTrustRegionClippingLossSeparateNetworksConfig:
    clipping_epsilon: float = 0.2
    value_clip_parameter: float = 0.2
    clip_value: bool = True
    entropy_cost: float = 0.01
    value_cost: float = 0.5


class MAPGWithTrustRegionClippingLossSeparateNetworks(Loss):
    def __init__(
        self,
        config: MAPGTrustRegionClippingLossSeparateNetworksConfig = MAPGTrustRegionClippingLossSeparateNetworksConfig(),  # noqa: E501
    ):
        """Component defines a MAPGWithTrustRegionClipping loss function.

        Specifically for a PPO system where the policy and critic networks
        are separate.

        Args:
            config : MAPGTrustRegionClippingLossSeparateNetworksConfig
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Create and store MAPGWithTrustRegionClippingLossSeparateNetworks loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def policy_loss_grad_fn(
            policy_params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios.

            Args:
                policy_params: policy network parameters.
                observations: agent observations.
                actions: actions the agents took.
                behaviour_log_probs: Log probabilities of actions taken by
                    current policy in the environment.
                advantages: advantage estimation values per agent.

            Returns:
                Tuple[policy gradients, policy loss information]
            """

            policy_grads = {}
            loss_info_policy = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def policy_loss_fn(
                    policy_params: Any,
                    observations: Any,
                    actions: jnp.ndarray,
                    behaviour_log_probs: jnp.ndarray,
                    advantages: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    """Inner policy loss function: see outer function for parameters."""
                    # TODO(Matthew): GNN application could go here?
                    print("\n\n\nOBS:\n", observations)
                    distribution_params = network.policy_network.apply(
                        policy_params, observations
                    )
                    log_probs = network.log_prob(distribution_params, actions)
                    entropy = network.entropy(distribution_params)
                    # Compute importance sampling weights:
                    # current policy / behavior policy.
                    rhos = jnp.exp(log_probs - behaviour_log_probs)
                    clipping_epsilon = self.config.clipping_epsilon

                    policy_loss = rlax.clipped_surrogate_pg_loss(
                        rhos, advantages, clipping_epsilon
                    )

                    # Entropy regulariser.
                    entropy_loss = -jnp.mean(entropy)

                    total_policy_loss = (
                        policy_loss + entropy_loss * self.config.entropy_cost
                    )

                    # TODO: (Ruan) Keeping the entropy penalty for now.
                    # can remove or add a flag for including it.
                    loss_info_policy = {
                        "policy_loss_total": total_policy_loss,
                        "loss_policy": policy_loss,
                        "loss_entropy": entropy_loss,
                    }

                    return total_policy_loss, loss_info_policy

                policy_grads[agent_key], loss_info_policy[agent_key] = jax.grad(
                    policy_loss_fn, has_aux=True
                )(
                    policy_params[agent_net_key],
                    observations[agent_key].observation,
                    actions[agent_key],
                    behaviour_log_probs[agent_key],
                    advantages[agent_key],
                )
            return policy_grads, loss_info_policy

        def critic_loss_grad_fn(
            critic_params: Any,
            observations: Any,
            target_values: Dict[str, jnp.ndarray],
            behavior_values: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Clipped critic loss.

            Args:
                critic_params: critic network parameters.
                observations: agent observations.
                actions: actions the agents took.
                target_values: target values to be used for optimizing the
                    critic network.
                behaviour_values: state values computed for observations
                    using the current critic network in the environment.

            Returns:
                Tuple[critic gradients, critic loss information]
            """

            critic_grads = {}
            loss_info_critic = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                def critic_loss_fn(
                    critic_params: Any,
                    observations: Any,
                    target_values: jnp.ndarray,
                    behavior_values: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    """Inner critic loss function: see outer function for parameters."""

                    values = network.critic_network.apply(critic_params, observations)

                    # Value function loss. Exclude the bootstrap value
                    unclipped_value_error = target_values - values
                    unclipped_value_loss = unclipped_value_error**2

                    value_clip_parameter = self.config.value_clip_parameter
                    if self.config.clip_value:
                        # Clip values to reduce variablility during critic training.

                        clipped_values = behavior_values + jnp.clip(
                            values - behavior_values,
                            -value_clip_parameter,
                            value_clip_parameter,
                        )
                        clipped_value_error = target_values - clipped_values
                        clipped_value_loss = clipped_value_error**2
                        value_loss = jnp.mean(
                            jnp.fmax(unclipped_value_loss, clipped_value_loss)
                        )
                    else:
                        value_loss = jnp.mean(unclipped_value_loss)

                    # TODO (Ruan): Including value loss parameter in the
                    # value loss for now but can add a flag
                    value_loss = value_loss * self.config.value_cost

                    loss_info_critic = {"loss_critic": value_loss}

                    return value_loss, loss_info_critic

                critic_grads[agent_key], loss_info_critic[agent_key] = jax.grad(
                    critic_loss_fn, has_aux=True
                )(
                    critic_params[agent_net_key],
                    observations[agent_key].observation,
                    target_values[agent_key],
                    behavior_values[agent_key],
                )
            return critic_grads, loss_info_critic

        # Save the gradient funcitons.
        trainer.store.policy_grad_fn = policy_loss_grad_fn
        trainer.store.critic_grad_fn = critic_loss_grad_fn

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGTrustRegionClippingLossSeparateNetworksConfig
