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
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal
from omegaconf import DictConfig

from mava.types import (
    Observation,
    ObservationGlobalState,
    RNNGlobalObservation,
    RNNObservation,
)


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation_fn: Callable[[chex.Array], chex.Array] = nn.relu
    use_layer_norm: bool = False

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


class FeedForwardActor(nn.Module):
    """Feedforward Actor Network."""

    torso: nn.Module
    num_actions: Sequence[int]

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.Categorical:
        """Forward pass."""
        x = observation.agents_view

        x = self.torso(x)
        actor_logits = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(x)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        return distrax.Categorical(logits=masked_logits)


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
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class RecurrentActor(nn.Module):
    """Recurrent Actor Network."""

    action_dim: Sequence[int]
    pre_torso: nn.Module
    post_torso: nn.Module

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, distrax.Categorical]:
        """Forward pass."""
        observation, done = observation_done

        policy_embedding = self.pre_torso(observation.agents_view)
        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN()(policy_hidden_state, policy_rnn_input)
        actor_logits = self.post_torso(policy_embedding)
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor_logits)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

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


def parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
    }
    return activation_fns[activation_fn_name]


def make(
    config: DictConfig, network: str, centralised_critic: bool = False
) -> Union[Tuple[FeedForwardActor, FeedForwardCritic], Tuple[RecurrentActor, RecurrentCritic]]:
    """Get the networks."""

    def create_torso(network_key: str, layer_size_key: str) -> MLPTorso:
        """Helper function to create a torso object from the config."""
        activation_fn = parse_activation_fn(config.network[network_key]["activation"])
        return MLPTorso(
            layer_sizes=config.network[network_key][layer_size_key],
            activation_fn=activation_fn,
            use_layer_norm=config.network[network_key].use_layer_norm,
        )

    if network == "feedforward":
        actor = FeedForwardActor(
            torso=create_torso("actor_network", "layer_sizes"),
            num_actions=config.system.num_actions,
        )
        critic = FeedForwardCritic(
            torso=create_torso("critic_network", "layer_sizes"),
            centralised_critic=centralised_critic,
        )
    elif network == "recurrent":
        actor = RecurrentActor(
            action_dim=config.system.num_actions,
            pre_torso=create_torso("actor_network", "pre_torso_layer_sizes"),
            post_torso=create_torso("actor_network", "post_torso_layer_sizes"),
        )
        critic = RecurrentCritic(
            pre_torso=create_torso("critic_network", "pre_torso_layer_sizes"),
            post_torso=create_torso("critic_network", "post_torso_layer_sizes"),
            centralised_critic=centralised_critic,
        )
    else:
        raise ValueError(f"The network '{network}' is not supported.")

    return actor, critic
