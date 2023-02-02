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
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Type

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import rlax
from chex import Array
from haiku._src.basic import merge_leading_dims

from mava.callbacks import Callback
from mava.components import Component, training
from mava.core_jax import SystemTrainer
from mava.utils.jax_training_utils import action_mask_categorical_policies


def clipped_surrogate_pg_loss(
    prob_ratios_t: Array,
    adv_t: Array,
    epsilon: float,
    mask: Array,
    use_stop_gradient: bool = True,
) -> Array:
    """Computes the clipped surrogate policy gradient loss.

    L_clipₜ(θ) = - min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)
    Where rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ) and Âₜ are the advantages.
    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347
    -----------------------------------------------
    This was copied from the RLAX packaged and only modifed to accept a mask
    -----------------------------------------------

    Args:
        prob_ratios_t:
            Ratio of action probabilities for actions a_t:
            rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ)
        adv_t:
            the observed or estimated advantages from executing actions a_t.
        epsilon:
            Scalar value corresponding to how much to clip the objecctive.
        mask:
            Mask to apply for sequence padding.
        use_stop_gradient:
            bool indicating whether or not to apply stop gradient to
            advantages.

    Returns:
        Loss whose gradient corresponds to a clipped surrogate policy gradient
            update.
    """
    chex.assert_rank([prob_ratios_t, adv_t], [1, 1])
    chex.assert_type([prob_ratios_t, adv_t], [float, float])

    adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
    clipped_ratios_t = jnp.clip(prob_ratios_t, 1.0 - epsilon, 1.0 + epsilon)
    clipped_objective = jnp.fmin(prob_ratios_t * adv_t, clipped_ratios_t * adv_t)

    eps = 1e-10

    return -jnp.sum(clipped_objective * mask) / (jnp.sum(mask) + eps)


class ValueLoss(Component):
    @abc.abstractmethod
    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """An abstract class for a component for the final value loss function."""

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "value_loss"


class SquaredErrorValueLoss(ValueLoss):
    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ):
        """Component defines a SquaredErrorValueLoss loss function.

        Args:
            config: SimpleNamespace.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Creates and stores the squared error value loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def squared_error_loss_fn(
            value_error: jax.interpreters.ad.JVPTracer,
        ) -> jax.interpreters.ad.JVPTracer:
            """Default loss function"""
            return value_error**2

        trainer.store.value_loss_fn = squared_error_loss_fn


@dataclass
class HuberValueLossConfig:
    huber_delta: float = 1.0


class HuberValueLoss(ValueLoss):
    def __init__(
        self,
        config: HuberValueLossConfig = HuberValueLossConfig(),
    ):
        """Component defines a HuberValueLoss loss function.

        Args:
            config: HuberValueLossConfig.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Creates and stores the huber value loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def huber_loss_fn(
            value_error: jax.interpreters.ad.JVPTracer,
        ) -> jax.interpreters.ad.JVPTracer:
            """Huber loss function"""
            return rlax.huber_loss(value_error, self.config.huber_delta)

        trainer.store.value_loss_fn = huber_loss_fn


class Loss(Component):
    @abc.abstractmethod
    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Abstract class component for the entire policy and critic loss functions."""

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
    """The value_clip_parameter should be relatively small when \
        value_normalization is True.

    The idea is to scale it to try and match the effect of the normalisation
    on the target values.
    """

    clipping_epsilon: float = 0.2
    value_clip_parameter: float = 0.2
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
        """Create and store MAPGWithTrustRegionClippingLoss loss function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def policy_loss_grad_fn(
            policy_params: Any,
            policy_states: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
            masks: Dict[str, jnp.ndarray],
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
                network = trainer.store.networks[agent_net_key]
                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.

                def policy_loss_fn(
                    policy_params: Any,
                    policy_states: Any,
                    observations: Any,
                    actions: jnp.ndarray,
                    behaviour_log_probs: jnp.ndarray,
                    advantages: jnp.ndarray,
                    mask: jnp.ndarray,
                    legals: Any,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    """Inner policy loss function: see outer function for \
                        parameters."""

                    # TODO (dries): Can we implement something more general here?
                    # Like a function call?
                    if policy_states:
                        # Recurrent actor.
                        minibatch_size = int(
                            trainer.store.epoch_batch_size
                            / trainer.store.num_minibatches
                        )
                        seq_len = trainer.store.sequence_length - 1

                        batch_seq_observations = observations.reshape(
                            minibatch_size, seq_len, -1
                        )

                        batch_seq_policy_states = policy_states[0].reshape(
                            minibatch_size, seq_len, -1
                        )

                        # Use the state at the start of the sequence
                        # and unroll the policy.
                        # core = lambda x, y: network.policy_network.apply(
                        #     policy_params, [x, y]
                        # )

                        def core(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                            return network.policy_network.apply(policy_params, [x, y])

                        distribution_params, _ = hk.static_unroll(
                            core,
                            batch_seq_observations,
                            batch_seq_policy_states[:, 0],
                            time_major=False,
                        )

                        # Flatten the distribution_params

                        distribution_params = jax.tree_util.tree_map(
                            lambda x: merge_leading_dims(x, 2),
                            distribution_params,
                        )
                    else:
                        # Feedforward actor.
                        distribution_params = network.policy_network.apply(
                            policy_params, observations
                        )

                    distribution_params = action_mask_categorical_policies(
                        distribution_params, legals
                    )
                    log_probs = network.log_prob(distribution_params, actions)
                    entropy = network.entropy(distribution_params)

                    # Compute importance sampling weights:
                    # current policy / behavior policy.
                    rhos = jnp.exp(log_probs - behaviour_log_probs)
                    clipping_epsilon = self.config.clipping_epsilon

                    policy_loss = clipped_surrogate_pg_loss(
                        rhos, advantages, clipping_epsilon, mask
                    )

                    eps = 1e-10
                    # Entropy regulariser.
                    entropy_loss = -jnp.sum(entropy * mask) / (jnp.sum(mask) + eps)

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
                    policy_states[agent_key],
                    observations[agent_key].observation,
                    actions[agent_key],
                    behaviour_log_probs[agent_key],
                    advantages[agent_key],
                    masks[agent_key],
                    observations[agent_key].legal_actions,
                )
            return policy_grads, loss_info_policy

        def critic_loss_grad_fn(
            critic_params: Any,
            observations: Any,
            target_values: Dict[str, jnp.ndarray],
            behavior_values: Dict[str, jnp.ndarray],
            masks: Dict[str, jnp.ndarray],
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
                network = trainer.store.networks[agent_net_key]

                def critic_loss_fn(
                    critic_params: Any,
                    observations: Any,
                    target_values: jnp.ndarray,
                    behavior_values: jnp.ndarray,
                    mask: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    """Inner critic loss function: see outer function for parameters."""

                    if network.get_critic_init_state() is not None:
                        minibatch_size = int(
                            trainer.store.epoch_batch_size
                            / trainer.store.num_minibatches
                        )
                        seq_len = trainer.store.sequence_length - 1

                        batch_seq_observations = observations.reshape(
                            minibatch_size, seq_len, -1
                        )

                        core = lambda x, y: network.critic_network.apply(  # noqa: E731
                            critic_params, [x, y]
                        )

                        values, _ = hk.static_unroll(
                            core,
                            batch_seq_observations,
                            network.get_critic_init_state(),
                            time_major=False,
                        )

                        values = jax.tree_util.tree_map(
                            lambda x: x.reshape((-1,) + x.shape[2:]),
                            values,
                        )

                    else:
                        values = (
                            network.critic_network.apply(critic_params, observations)
                            * mask
                        )

                    # Value function loss. Exclude the bootstrap value
                    unclipped_value_error = target_values - values

                    unclipped_value_loss = trainer.store.value_loss_fn(
                        unclipped_value_error
                    )

                    eps = 1e-10
                    value_clip_parameter = self.config.value_clip_parameter
                    if self.config.clip_value:
                        # Clip values to reduce variablility during critic training.

                        clipped_values = behavior_values + jnp.clip(
                            values - behavior_values,
                            -value_clip_parameter,
                            value_clip_parameter,
                        )
                        clipped_value_error = target_values - clipped_values
                        clipped_value_loss = trainer.store.value_loss_fn(
                            clipped_value_error
                        )
                        value_loss = jnp.sum(
                            jnp.fmax(unclipped_value_loss, clipped_value_loss)
                        ) / (jnp.sum(mask) + eps)
                    else:
                        value_loss = jnp.sum(unclipped_value_loss) / (
                            jnp.sum(mask) + eps
                        )

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
                    masks[agent_key],
                )
            return critic_grads, loss_info_critic

        # Save the gradient funcitons.
        trainer.store.policy_grad_fn = policy_loss_grad_fn
        trainer.store.critic_grad_fn = critic_loss_grad_fn
