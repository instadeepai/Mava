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

from typing import Optional, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal

from mava.networks.retention import MultiScaleRetention
from mava.utils.networks_util import SwiGLU


class EncodeBlock(nn.Module):
    """Sable encoder block."""

    embed_dim: int
    n_head: int
    n_agent: int
    decay_scaling_factor: float

    def setup(self) -> None:
        # Initialize the RMSNorm layer normalization
        self.ln1 = nn.RMSNorm()
        self.ln2 = nn.RMSNorm()

        # Initialize the MultiScaleRetention
        self.retn = MultiScaleRetention(
            embed_dim=self.embed_dim,
            n_head=self.n_head,
            n_agent=self.n_agent,
            full_self_retention=True,  # Full retention for the encoder
            decay_scaling_factor=self.decay_scaling_factor,
        )

        # Initialize SwiGLU feedforward network
        self.ffn = SwiGLU(self.embed_dim, self.embed_dim)

    def __call__(self, x: chex.Array) -> chex.Array:
        """Applies Parallel MultiScaleRetention."""
        x = self.ln1(x + self.retn(key=x, query=x, value=x))
        output = self.ln2(x + self.ffn(x))
        return output

    def recurrent(self, x: chex.Array, hstate: chex.Array) -> chex.Array:
        """Applies Recurrent MultiScaleRetention."""
        ret, updated_hstate = self.retn.recurrent(key_n=x, query_n=x, value_n=x, hstate=hstate)
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate

    def chunkwise(self, x: chex.Array, hstate: chex.Array, dones: chex.Array) -> chex.Array:
        """Applies Chunkwise MultiScaleRetention."""
        ret, updated_hstate = self.retn.chunkwise(
            key=x, query=x, value=x, hstate=hstate, dones=dones
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate


class Encoder(nn.Module):
    """Multi-block encoder consisting of multiple `EncoderBlock` modules."""

    n_block: int
    embed_dim: int
    n_head: int
    n_agent: int
    decay_scaling_factor: float = 1.0

    def setup(self) -> None:
        # Initialize the RMSNorm layer normalization
        self.ln = nn.RMSNorm()

        # Initialize the observation encoder and value head layers
        self.obs_encoder = nn.Sequential(
            [
                nn.RMSNorm(),
                nn.Dense(self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2)), use_bias=False),
                nn.gelu,
            ],
        )
        self.head = nn.Sequential(
            [
                nn.Dense(self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.RMSNorm(),
                nn.Dense(1, kernel_init=orthogonal(0.01)),
            ],
        )

        # Initialize the encoder blocks
        self.blocks = [
            EncodeBlock(
                self.embed_dim,
                self.n_head,
                self.n_agent,
                self.decay_scaling_factor,
                name=f"encoder_block_{block_id}",
            )
            for block_id in range(self.n_block)
        ]

    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Apply parallel encoding (default)."""
        # Encode the observation
        obs_rep = self.obs_encoder(obs)

        # Apply the encoder blocks
        for block in self.blocks:
            obs_rep = block(self.ln(obs_rep))

        # Compute the value function
        v_loc = self.head(obs_rep)

        return v_loc, obs_rep

    def recurrent(
        self, obs: chex.Array, hstate: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply recurrent encoding."""
        # Initialize the updated hidden state
        updated_hstate = jnp.zeros_like(hstate)
        # Encode the observation
        obs_rep = self.obs_encoder(obs)

        # Apply the encoder blocks
        for i, block in enumerate(self.blocks):
            # Get the hidden state for the current block
            hs = hstate[:, :, i]
            # Apply the recurrent encoder block
            obs_rep, hs_new = block.recurrent(self.ln(obs_rep), hs)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)

        # Compute the value function
        v_loc = self.head(obs_rep)

        return v_loc, obs_rep, updated_hstate

    def chunkwise(
        self, obs: chex.Array, hstate: chex.Array, dones: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply chunkwise encoding."""
        # Initialize the updated hidden state
        updated_hstate = jnp.zeros_like(hstate)
        # Encode the observation
        obs_rep = self.obs_encoder(obs)

        # Apply the encoder blocks
        for i, block in enumerate(self.blocks):
            # Get the hidden state for the current block
            hs = hstate[:, :, i]
            # Apply the chunkwise encoder block
            obs_rep, hs_new = block.chunkwise(self.ln(obs_rep), hs, dones)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)

        # Compute the value function
        v_loc = self.head(obs_rep)

        return v_loc, obs_rep, updated_hstate


class DecodeBlock(nn.Module):
    """Sable decoder block."""

    embed_dim: int
    n_head: int
    n_agent: int
    decay_scaling_factor: float
    use_swiglu: bool = False

    def setup(self) -> None:
        # Initialize the RMSNorm layer normalization
        self.ln1, self.ln2, self.ln3 = nn.RMSNorm(), nn.RMSNorm(), nn.RMSNorm()

        # Initialize the MultiScaleRetention
        self.retn1 = MultiScaleRetention(
            embed_dim=self.embed_dim,
            n_head=self.n_head,
            n_agent=self.n_agent,
            full_self_retention=False,  # Masked retention for the decoder
            decay_scaling_factor=self.decay_scaling_factor,
        )
        self.retn2 = MultiScaleRetention(
            embed_dim=self.embed_dim,
            n_head=self.n_head,
            n_agent=self.n_agent,
            full_self_retention=False,  # Masked retention for the decoder
            decay_scaling_factor=self.decay_scaling_factor,
        )

        # Initialize SwiGLU feedforward network
        self.ffn = SwiGLU(self.embed_dim, self.embed_dim)

    def __call__(self, x: chex.Array, obs_rep: chex.Array) -> chex.Array:
        """Applies Parallel MultiScaleRetention."""
        x = self.ln1(x + self.retn1(key=x, query=x, value=x))
        x = self.ln2(obs_rep + self.retn2(key=x, query=obs_rep, value=x))
        output = self.ln3(x + self.ffn(x))
        return output

    def recurrent(
        self,
        x: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Recurrent MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1.recurrent(key_n=x, query_n=x, value_n=x, hstate=hs1)
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2.recurrent(key_n=ret, query_n=obs_rep, value_n=ret, hstate=hs2)
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)

    def chunkwise(
        self,
        x: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        dones: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Chunkwise MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1.chunkwise(key=x, query=x, value=x, hstate=hs1, dones=dones)
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2.chunkwise(
            key=ret,
            query=obs_rep,
            value=ret,
            hstate=hs2,
            dones=dones,
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)


class Decoder(nn.Module):
    """Multi-block decoder consisting of multiple `DecoderBlock` modules."""

    n_block: int
    embed_dim: int
    n_head: int
    n_agent: int
    decay_scaling_factor: float = 1.0
    action_space_type: str = "discrete"

    def setup(self) -> None:
        # Initialize the RMSNorm layer normalization
        self.ln = nn.RMSNorm()

        # Initialize action encoder based on action space type
        if self.action_space_type == "discrete":
            self.action_encoder = nn.Sequential(
                [
                    nn.Dense(self.n_embd, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                ],
            )
            # Set log_std to None for discrete action spaces as it is not used
            self.log_std = None
        else:
            self.action_encoder = nn.Sequential(
                [nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
            )
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        # Initialize the head layer for the action logits
        self.head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.RMSNorm(),
                nn.Dense(self.action_dim, kernel_init=orthogonal(0.01)),
            ],
        )

        # Initialize the decoder blocks
        self.blocks = [
            DecodeBlock(
                self.embed_dim,
                self.n_head,
                self.n_agent,
                self.decay_scaling_factor,
                name=f"decoder_block_{block_id}",
            )
            for block_id in range(self.n_block)
        ]

    def __call__(self, action: chex.Array, obs_rep: chex.Array) -> chex.Array:
        """Apply parallel decoding (default)."""
        # Encode the action
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Apply the decoder blocks
        for block in self.blocks:
            x = block.parallel(x, obs_rep)

        # Compute the action logits
        logit = self.head(x)

        return logit

    def recurrent(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Apply recurrent decoding."""
        # Initialize the updated hidden states
        updated_hstates = (jnp.zeros_like(hstates[0]), jnp.zeros_like(hstates[1]))
        # Encode the action
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = jax.tree.map(lambda x, i=i: x[:, :, i], hstates)
            x, hs_new = block.recurrent(x=x, obs_rep=obs_rep, hstates=hs)
            updated_hstates = jax.tree.map(
                lambda x, y: x.at[:, :, i].set(y), updated_hstates, hs_new
            )

        # Compute the action logits
        logit = self.head(x)

        return logit, updated_hstates

    def chunkwise(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        dones: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Apply chunkwise decoding."""
        # Initialize the updated hidden states
        updated_hstates = (jnp.zeros_like(hstates[0]), jnp.zeros_like(hstates[1]))
        # Encode the action
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = jax.tree.map(lambda x, i=i: x[:, :, i], hstates)
            x, hs_new = block.chunkwise(x=x, obs_rep=obs_rep, hstates=hs, dones=dones)
            updated_hstates = jax.tree.map(
                lambda x, y: x.at[:, :, i].set(y), updated_hstates, hs_new
            )

        # Compute the action logits
        logit = self.head(x)

        return logit, updated_hstates


class SableNetwork(nn.Module):
    """Sable network module."""

    n_block: int
    embed_dim: int
    n_head: int
    n_agent: int
    decay_scaling_factor: float = 1.0
    action_space_type: str = "discrete"

    def setup(self) -> None:
        # Check if the action space type is valid
        assert self.action_space_type in [
            "discrete",
            "continuous",
        ], "Invalid action space type"

        self.encoder = Encoder(
            self.n_block,
            self.embed_dim,
            self.n_head,
            self.n_agent,
            self.decay_scaling_factor,
        )
        self.decoder = Decoder(
            self.n_block,
            self.embed_dim,
            self.n_head,
            self.n_agent,
            self.decay_scaling_factor,
            self.action_space_type,
        )
        if self.action_space_type == "discrete":
            # TODO: add chunkwise
            self.autoregressive_act = discrete_autoregressive_act
            self.parallel_act = discrete_parallel_act
        # TODO: Implement continuous executor and trainer
        # else:
        #    self.executor = ContinuousExecutor()
        #    self.trainer = ContinuousTrainer()

        # Decay kappa for each head
        self.decay_kappas = 1 - jnp.exp(
            jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), self.n_head)
        )
        self.decay_kappas = self.decay_kappas * self.decay_scaling_factor
        if self.decay_scaling_factor <= 0:
            self.decay_kappas = jnp.ones_like(
                self.decay_kappas
            )  # No decaying if decay_scaling_factor is 0

    def __call__(
        self,
        obs: chex.Array,
        action: chex.Array,
        legal_actions: chex.Array,
        hstates: Tuple[chex.Array, Tuple[chex.Array, chex.Array]],  # TODO: make hstates named tuple
        dones: chex.Array,
        rng_key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Training phase."""
        # Apply the encoder
        v_loc, obs_rep, _ = self.encoder.chunkwise(obs=obs, hstate=hstates[0], dones=dones)
        # Apply the decoder
        action_log, entropy = self.parallel_act(
            decoder=self.decoder,
            n_agent=self.n_agent,
            obs_rep=obs_rep,
            action=action,
            legal_actions=legal_actions,
            hstates=hstates[1],
            dones=dones,
            rng_key=rng_key,
        )

        return action_log, v_loc, entropy

    def get_actions(
        self,
        obs: chex.Array,
        legal_actions: chex.Array,
        hstates: Tuple[chex.Array, Tuple[chex.Array, chex.Array]],
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, Tuple[chex.Array, chex.Array]]:
        """Inference phase."""
        # Decay the hidden states
        decayed_hstates = jax.tree.map(
            lambda x: x * self.decay_kappa[None, :, None, None, None], hstates
        )
        # Apply the encoder
        v_loc, obs_rep, updated_enc_hs = self.encoder.recurrent(obs, decayed_hstates[0])

        # Apply the decoder
        output_actions, output_actions_log, updated_dec_hs = self.autoregressive_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            legal_actions=legal_actions,
            hstates=decayed_hstates[1],
            key=key,
        )

        # TODO: find a way to remove raw action from the output
        # Always returning None for the raw action now since we will never use it for Sable.
        # We still need to do this though since this is the way the networks work for MAT and
        # changing this will break the evaluator API.
        return (
            output_actions,
            output_actions_log,
            v_loc,
            None,
            (updated_enc_hs, updated_dec_hs),
        )


def discrete_parallel_act(
    decoder: Decoder,
    n_agent: int,
    obs_rep: chex.Array,
    action: chex.Array,
    legal_actions: chex.Array,
    hstates: chex.Array,
    dones: chex.Array,
    rng_key: Optional[chex.PRNGKey] = None,
) -> Tuple[chex.Array, chex.Array]:
    """Parallel action sampling for discrete action spaces."""
    # Delete `rng_key` since it is not used in discrete action space
    del rng_key
    # Get the batch size, sequence length, and action dimension
    batch_size, sequence_size, action_dim = action.shape

    # Create a shifted action sequence for predicting the next action
    # Initialize the shifted action sequence.
    shifted_actions = jnp.zeros((batch_size, sequence_size, action_dim + 1))

    # Set the start-of-timestep token (first action as a "start" signal)
    start_timestep_token = jnp.zeros(action_dim + 1).at[0].set(1)

    # One hot encode the action
    one_hot_action = jax.nn.one_hot(action, action_dim)

    # Insert one-hot encoded actions into shifted array, shifting by 1 position
    shifted_actions = shifted_actions.at[:, :, 1:].set(one_hot_action)
    shifted_actions = jnp.roll(shifted_actions, shift=1, axis=1)

    # Set the start token for the first agent in each timestep
    shifted_actions = shifted_actions.at[:, ::n_agent, :].set(start_timestep_token)

    # Apply the decoder
    logit, _ = decoder.chunkwise(
        action=shifted_actions,
        obs_rep=obs_rep,
        hstates=hstates,
        dones=dones,
    )

    # Mask the logits for illegal actions
    masked_logits = jnp.where(
        legal_actions,
        logit,
        jnp.finfo(jnp.float32).min,
    )

    # Create a categorical distribution over the masked logits
    distribution = distrax.Categorical(logits=masked_logits)

    # Compute the log probability of the actions
    action_log_prob = distribution.log_prob(action)
    action_log_prob = jnp.expand_dims(action_log_prob, axis=-1)

    # Compute the entropy of the action distribution
    entropy = jnp.expand_dims(distribution.entropy(), axis=-1)

    return action_log_prob, entropy


def discrete_autoregressive_act(
    decoder: Decoder,
    obs_rep: chex.Array,
    hstates: chex.Array,
    legal_actions: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    # Get the batch size, sequence length, and action dimension
    batch_size, n_agent, action_dim = legal_actions.shape

    # Create a shifted action sequence for predicting the next action
    # Initialize the shifted action sequence.
    shifted_actions = jnp.zeros((batch_size, n_agent, action_dim + 1))
    # Set the start-of-timestep token (first action as a "start" signal)
    shifted_actions = jnp.zeros(action_dim + 1).at[0].set(1)

    # Define the output action and output action log sizes
    output_action = jnp.zeros((batch_size, n_agent, 1))
    output_action_log = jnp.zeros_like(output_action)

    # Apply the decoder autoregressively
    for i in range(n_agent):
        logit, updated_hstates = decoder.recurrent(
            action=shifted_actions[:, i : i + 1, :],
            obs_rep=obs_rep[:, i : i + 1, :],
            hstates=hstates,
        )
        # Mask the logits for illegal actions
        masked_logits = jnp.where(
            legal_actions[:, i : i + 1, :],
            logit,
            jnp.finfo(jnp.float32).min,
        )
        # Create a categorical distribution over the masked logits
        distribution = distrax.Categorical(logits=masked_logits)
        # Sample an action from the distribution
        key, sample_key = jax.random.split(key)
        action, action_log = distribution.sample_and_log_prob(seed=sample_key)
        # Set the action and action log
        output_action = output_action.at[:, i, :].set(action)
        output_action_log = output_action_log.at[:, i, :].set(action_log)

        # Update the shifted action
        update_shifted_action = i + 1 < n_agent
        shifted_actions = jax.lax.cond(
            update_shifted_action,
            lambda action=action, i=i, shifted_actions=shifted_actions: shifted_actions.at[
                :, i + 1, 1:
            ].set(jax.nn.one_hot(action[:, 0], action_dim)),
            lambda shifted_actions=shifted_actions: shifted_actions,
        )

    return output_action.astype(jnp.int32), output_action_log, updated_hstates
