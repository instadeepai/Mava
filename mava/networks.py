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
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal, lecun_normal

from mava.distributions import IdentityTransformation, TanhTransformedDistribution
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
    kernel_init: str = "orthogonal" # orthogonal or lecun_normal
    use_layer_norm: bool = False
    activate_final: bool = True

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)
        self.kernel_init_fn = _parse_kernel_init_fn(self.kernel_init)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(layer_size, kernel_init=self.kernel_init_fn)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
                
            if i != len(self.layer_sizes) - 1: 
                x = self.activation_fn(x)
            elif i == len(self.layer_sizes) - 1 and self.activate_final:
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

    action_dim: int

    @nn.compact
    def __call__(
        self, obs_embedding: chex.Array, observation: Observation
    ) -> tfd.TransformedDistribution:
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

        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(obs_embedding)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        #  We transform this distribution with the `Identity()` transformation to
        # keep the API identical to the ContinuousActionHead.
        return IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))


class ContinuousActionHead(nn.Module):
    """ContinuousActionHead using a transformed Normal distribution.

    Note: This network only handles the case where actions lie in the interval [-1, 1].
    """

    action_dim: int
    min_scale: float = 1e-3

    def setup(self) -> None:
        self.mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))
        self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

    @nn.compact
    def __call__(self, obs_embedding: chex.Array, observation: Observation) -> tfd.Independent:
        """Action selection for continuous action space environments.

        Args:
            obs_embedding (chex.Array): Observation embedding.
            observation (Observation): Observation object.

        Returns:
            tfd.Independent: Independent transformed distribution.
        """
        loc = self.mean(obs_embedding)
        scale = jax.nn.softplus(self.log_std) + self.min_scale
        distribution = tfd.Normal(loc=loc, scale=scale)

        return tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
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

def _parse_kernel_init_fn(kernel_init_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get kernel init function."""
    init_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "orthogonal": orthogonal(np.sqrt(2)),
        "lecun_normal": lecun_normal(),
    }
    return init_fns[kernel_init_fn_name]