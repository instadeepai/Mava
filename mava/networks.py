import functools
from typing import Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
from mava.types import RNNObservation, Observation
from omegaconf import DictConfig
from chex import Array

class Torso(nn.Module):
    """Feedforward torso."""
    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False

    def setup(self) -> None:
        if self.activation == "relu":
            self.activation_fn = nn.relu
        elif self.activation == "tanh":
            self.activation_fn = nn.tanh
    @nn.compact
    def __call__(self, observation: Array):
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)
        return x


class FF_Actor(nn.Module):
    """Feedforward Actor Network."""

    torso: nn.Module
    num_actions: Sequence[int]

    @nn.compact
    def __call__(self, observation):
        x = observation.agents_view
        x = self.torso(x)
        actor_output = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        
        masked_logits = jnp.where(
            observation.action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )
        actor_policy = distrax.Categorical(logits=masked_logits)

        return actor_policy
    
class FF_Critic(nn.Module):
    """Feedforward Critic Network."""

    torso: nn.Module
    centralized_critic: bool = False

    @nn.compact
    def __call__(self, observation):
        """Forward pass."""
        if self.centralized_critic:
            observation = observation.global_state
        else:
            observation = observation.agents_view
        critic_output = self.torso(observation)
        critic_output = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic_output)
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


class Rec_Actor(nn.Module):
    """Recurrent Actor Network."""

    action_dim: Sequence[int]
    pre_torso: nn.Module
    post_torso: nn.Module

    @nn.compact
    def __call__(self, policy_hidden_state, observation_done):
        """Forward pass."""
        observation, done = observation_done

        policy_embedding = self.pre_torso(observation.agents_view)

        policy_rnn_in = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN()(policy_hidden_state, policy_rnn_in)

        actor_output = self.post_torso(policy_embedding)

        actor_output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)
        
        masked_logits = jnp.where(
            observation.action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        return policy_hidden_state, pi


class Rec_Critic(nn.Module):
    """Recurrent Critic Network."""
    
    pre_torso: nn.Module
    post_torso: nn.Module
    centralized_critic: bool = False

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done

        if self.centralized_critic:
            observation = observation.global_state
        else:
            observation = observation.agents_view
        critic_embedding = self.pre_torso(observation)

        critic_rnn_in = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN()(critic_hidden_state, critic_rnn_in)

        critic_output = self.post_torso(critic_embedding)
        critic_output = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic_output)

        return critic_hidden_state, jnp.squeeze(critic_output, axis=-1)


def get_networks(config: DictConfig, network: str, centralized_critic: bool =False)->Union[Tuple[FF_Actor, FF_Critic], Tuple[Rec_Actor, Rec_Critic]]:
    """Get the networks."""
    if network == "feedforward":
        actor = FF_Actor(
            torso=Torso(
                layer_sizes=config["system"]["actor_network"]["layer_sizes"],
                activation=config["system"]["actor_network"]["activation"],
                use_layer_norm=config["system"]["actor_network"]["use_layer_norm"],
            ),
            num_actions=config["system"]["num_actions"],
        )
        critic = FF_Critic(
            torso=Torso(
                layer_sizes=config["system"]["critic_network"]["layer_sizes"],
                activation=config["system"]["critic_network"]["activation"],
                use_layer_norm=config["system"]["critic_network"]["use_layer_norm"],
            ),
            centralized_critic=centralized_critic,
        )
    elif network == "recurrent":
        actor = Rec_Actor(
            action_dim=config["system"]["num_actions"],
            pre_torso=Torso(
                layer_sizes=config["system"]["actor_network"]["pre_torso_layer_sizes"],
                activation=config["system"]["actor_network"]["activation"],
                use_layer_norm=config["system"]["actor_network"]["use_layer_norm"],
            ),
            post_torso=Torso(
                layer_sizes=config["system"]["actor_network"]["post_torso_layer_sizes"],
                activation=config["system"]["actor_network"]["activation"],
                use_layer_norm=config["system"]["actor_network"]["use_layer_norm"],
            ),
        )
        critic = Rec_Critic(
            pre_torso=Torso(
                layer_sizes=config["system"]["critic_network"]["pre_torso_layer_sizes"],
                activation=config["system"]["critic_network"]["activation"],
                use_layer_norm=config["system"]["critic_network"]["use_layer_norm"],
            ),
            post_torso=Torso(
                layer_sizes=config["system"]["critic_network"]["post_torso_layer_sizes"],
                activation=config["system"]["critic_network"]["activation"],
                use_layer_norm=config["system"]["critic_network"]["use_layer_norm"],
            ),
            centralized_critic=centralized_critic,
        )
    else:
        raise ValueError(f"Network {config['system']['network']} not supported.")
    
    return actor, critic