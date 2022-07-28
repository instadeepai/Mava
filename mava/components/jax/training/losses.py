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
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import rlax

from mava.components.jax.training.base import Loss
from mava.core_jax import SystemTrainer


@dataclass
class MAPGTrustRegionClippingLossConfig:
    clipping_epsilon: float = 0.2
    clip_value: bool = True
    entropy_cost: float = 0.01
    value_cost: float = 0.5
    use_adaptive_entropy: bool = False


class MAPGWithTrustRegionClippingLoss(Loss):
    def __init__(
        self,
        config: MAPGTrustRegionClippingLossConfig = MAPGTrustRegionClippingLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def loss_grad_fn(
            params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
            behavior_values: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios."""

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

                    # For adaptive entropy
                    # https://arxiv.org/pdf/2007.02529.pdf - LICA
                    use_adaptive_entropy = self.config.use_adaptive_entropy
                    if use_adaptive_entropy:
                        logits = distribution_params.parameters["logits"]
                        probs = jax.nn.softmax(logits)
                        log_information = jax.lax.log(probs)
                        normalising_term = jax.numpy.linalg.norm(
                            log_information, axis=-1
                        )
                        adaptive_entropy = -jnp.mean(normalising_term)
                        entropy_loss = entropy_loss / adaptive_entropy

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
    use_adaptive_entropy: bool = False


class MAPGWithTrustRegionClippingLossSeparateNetworks(Loss):
    def __init__(
        self,
        config: MAPGTrustRegionClippingLossSeparateNetworksConfig = MAPGTrustRegionClippingLossSeparateNetworksConfig(),  # noqa: E501
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def policy_loss_grad_fn(
            policy_params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios."""

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

                    # For adaptive entropy
                    # https://arxiv.org/pdf/2007.02529.pdf - LICA
                    use_adaptive_entropy = self.config.use_adaptive_entropy
                    if use_adaptive_entropy:
                        logits = distribution_params.parameters["logits"]
                        probs = jax.nn.softmax(logits)
                        log_information = jax.lax.log(probs)
                        normalising_term = jax.numpy.linalg.norm(
                            log_information, axis=-1
                        )
                        adaptive_entropy = -jnp.mean(normalising_term)
                        entropy_loss = entropy_loss / adaptive_entropy

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
            """Surrogate loss using clipped probability ratios."""

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

                    values = network.critic_network.apply(critic_params, observations)

                    # values = jnp.squeeze(values, axis=-1)

                    # Value function loss. Exclude the bootstrap value
                    unclipped_value_error = target_values - values
                    unclipped_value_loss = unclipped_value_error**2

                    value_clip_parameter = self.config.value_clip_parameter
                    if self.config.clip_value:
                        # Clip values to reduce variablility during critic training.
                        # TODO (Ruan): Updated the value clipping fix this
                        # normal implementation as well check that this is also
                        # done in TF implementation
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


@dataclass
class MAPPOLossNetworksConfig:
    clipping_epsilon: float = 0.2
    value_clip_parameter: float = 0.2
    clip_value: bool = True
    entropy_cost: float = 0.01
    value_cost: float = 0.5
    use_adaptive_entropy: bool = False


class MAPPOLossSeparateNetworks(Loss):
    def __init__(
        self,
        config: MAPPOLossNetworksConfig = MAPPOLossNetworksConfig(),  # noqa: E501
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def policy_loss_grad_fn(
            policy_params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios."""

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

                    # For adaptive entropy
                    # https://arxiv.org/pdf/2007.02529.pdf - LICA
                    use_adaptive_entropy = self.config.use_adaptive_entropy
                    if use_adaptive_entropy:
                        logits = distribution_params.parameters["logits"]
                        probs = jax.nn.softmax(logits)
                        log_information = jax.lax.log(probs)
                        normalising_term = jax.numpy.linalg.norm(
                            log_information, axis=-1
                        )
                        adaptive_entropy = -jnp.mean(normalising_term)
                        entropy_loss = entropy_loss / adaptive_entropy

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
            """Surrogate loss using clipped probability ratios."""

            all_obs_list = []
            for key in trainer.store.trainer_agents:
                # agent_keys.append(key)
                all_obs_list.append(observations[key].observation)

            joint_obs = jax.numpy.concatenate(all_obs_list, axis=-1)

            agent_net_key = trainer.store.trainer_agent_net_keys["agent_0"]
            network = trainer.store.networks["networks"][agent_net_key]

            def critic_loss_fn(
                critic_params: Any,
                joint_obs: Any,
                target_values: jnp.ndarray,
                behavior_values: jnp.ndarray,
            ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                value_clip_parameter = self.config.value_clip_parameter
                values = network.critic_network.apply(critic_params, joint_obs)
                unclipped_value_error = target_values - values
                unclipped_value_loss = unclipped_value_error**2
                if self.config.clip_value:
                    # Clip values to reduce variablility during critic training.
                    # TODO (Ruan): Updated the value clipping fix this
                    # normal implementation as well check that this is also
                    # done in TF implementation
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

                    # TODO (Ruan): Including value loss parameter in the
                    # value loss for now but can add a flag
                    value_loss = value_loss * self.config.value_cost

                    loss_info_critic = {"loss_critic": value_loss}

                else:
                    value_loss = jnp.mean(unclipped_value_loss)
                return value_loss, loss_info_critic

            critic_grad, loss_info_critic_0 = jax.grad(critic_loss_fn, has_aux=True)(
                critic_params[agent_net_key],
                joint_obs,
                target_values["agent_0"],
                behavior_values["agent_0"],
            )

            critic_grads = {}
            loss_info_critic = {}
            for agent_key in trainer.store.trainer_agents:
                if agent_key == "agent_0":
                    critic_grads[agent_key] = critic_grad
                else:
                    critic_grads[agent_key] = jax.lax.stop_gradient(critic_grad)
                loss_info_critic[agent_key] = loss_info_critic_0

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
        return MAPPOLossNetworksConfig
