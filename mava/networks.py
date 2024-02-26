# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

import functools
from typing import Callable, Dict, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal
from jumanji.env import Environment

from mava.types import (
    Observation,
    ObservationGlobalState,
    RNNGlobalObservation,
    RNNObservation,
)


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(layer_size, kernel_init=orthogonal(np.sqrt(2)))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)
        return x


class CNNTorso(nn.Module):
    """CNN torso."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for channel, kernel, stride in zip(self.channel_sizes, self.kernel_sizes, self.strides):
            x = nn.Conv(channel, (kernel, kernel), (stride, stride))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)

        return x.reshape((x.shape[0], -1))


class DiscreteActionHead(nn.Module):
    """Discrete Action Head"""

    env: Environment
    sigma_min: float
    sigma_max: float

    @nn.compact
    def __call__(self, obs_embedding: chex.Array, observation: Observation) -> tfd.Categorical:
        """Action selection for distrete action space environments.

        Args:
            obs_embedding: Observation embedding from network torso.
            observation: Observation object containing `agents_view`, `action_mask` and
                `step_count`.

        Returns:
            A transformed tfd.categorical distribution on the action space for action sampling.

        NOTE: We pass both the observation embedding and the observation object to the action head
        since the observation object contains the action mask and other potentially useful
        information.
        """

        actor_logits = nn.Dense(self.env.action_dim, kernel_init=orthogonal(0.01))(obs_embedding)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        #  We transform this distribution with the `Identity()` transformation to enable us to call
        # `pi.distribution.entropy()` and keep the API identical to the ContinuousActionHead.
        return tfd.TransformedDistribution(
            distribution=tfd.Categorical(logits=masked_logits), bijector=tfb.Identity()
        )


class ContinuousActionHead(nn.Module):
    """Continuous Action Head"""

    env: Environment
    sigma_min: float
    sigma_max: float

    def setup(self) -> None:
        self.actor_mean = nn.Dense(self.env.action_dim, kernel_init=orthogonal(0.01))
        self.actor_log_std = self.param("log_std", nn.initializers.zeros, (self.env.action_dim,))
        min_action, max_action = self.env.min_max_action
        scale = 0.5 * (max_action - min_action)
        self.bijector = tfb.Chain(
            [tfb.Shift(min_action), tfb.Scale(scale=scale), tfb.Shift(1), tfb.Tanh()]
        )

    def _sigma_activation(self, sigma: chex.Array) -> chex.Array:
        """Don't allow sigma to grow too large for numerical stability."""
        return jnp.exp(
            self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (jnp.tanh(sigma) + 1.0)
        )

    @nn.compact
    def __call__(
        self, obs_embedding: chex.Array, observation: Observation
    ) -> tfd.MultivariateNormalDiag:
        """Action selection for continuous action space environments.

        Args:
            obs_embedding: Observation embedding from network torso.
            observation: Observation object containing `agents_view`, `action_mask` and
                `step_count`.

        Returns:
            A transformer tfd.MultivariateNormalDiag distribution.
        """

        mean = self.actor_mean(obs_embedding)
        std = self._sigma_activation(self.actor_log_std)

        return tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=mean, scale_diag=std),
            bijector=self.bijector,
        )


class FeedForwardActor(nn.Module):
    """Feed Forward Actor Network."""

    torso: nn.Module
    action_head: nn.Module

    @nn.compact
    def __call__(self, observation: Observation) -> tfd.Distribution:
        """Forward pass."""

        obs_embedding = self.torso(observation.agents_view)

        return self.action_head(obs_embedding, observation)


class FeedForwardCritic(nn.Module):
    """Feedforward Critic Network."""

    torso: nn.Module
    centralised_critic: bool = False

    @nn.compact
    def __call__(self, observation: Union[Observation, ObservationGlobalState]) -> chex.Array:
        """Forward pass."""
        if self.centralised_critic:
            if not isinstance(observation, ObservationGlobalState):
                raise ValueError("Global state must be provided to the centralised critic.")
            # Get global state in the case of a centralised critic.
            observation = observation.global_state
        else:
            # Get single agent view in the case of a decentralised critic.
            observation = observation.agents_view

        critic_output = self.torso(observation)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0))(critic_output)

        return jnp.squeeze(critic_output, axis=-1)


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, :, jnp.newaxis],
            self.initialize_carry((ins.shape[0], ins.shape[1]), ins.shape[2]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: Sequence[int], hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size))


class RecurrentActor(nn.Module):
    """Recurrent Actor Network."""

    pre_torso: nn.Module
    post_torso: nn.Module
    action_head: nn.Module

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        """Forward pass."""
        observation, done = observation_done

        policy_embedding = self.pre_torso(observation.agents_view)
        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN()(policy_hidden_state, policy_rnn_input)
        policy_embedding = self.post_torso(policy_embedding)
        pi = self.action_head(policy_embedding, observation)

        return policy_hidden_state, pi


class RecurrentCritic(nn.Module):
    """Recurrent Critic Network."""

    pre_torso: nn.Module
    post_torso: nn.Module
    centralised_critic: bool = False

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: Union[RNNObservation, RNNGlobalObservation],
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done

        if self.centralised_critic:
            if not isinstance(observation, ObservationGlobalState):
                raise ValueError("Global state must be provided to the centralised critic.")
            # Get global state in the case of a centralised critic.
            observation = observation.global_state
        else:
            # Get single agent view in the case of a decentralised critic.
            observation = observation.agents_view

        critic_embedding = self.pre_torso(observation)
        critic_rnn_input = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN()(critic_hidden_state, critic_rnn_input)
        critic_output = self.post_torso(critic_embedding)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0))(critic_output)

        return critic_hidden_state, jnp.squeeze(critic_output, axis=-1)


def _parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
    }
    return activation_fns[activation_fn_name]
