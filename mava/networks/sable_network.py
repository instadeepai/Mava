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

from functools import partial
from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal
from omegaconf import DictConfig

from mava.networks.retention import MultiScaleRetention
from mava.networks.utils.sable.discrete_trainer_executor import *  # noqa
from mava.systems.sable.types import HiddenStates
from mava.types import Observation
from mava.utils.sable_utils import SwiGLU


class EncodeBlock(nn.Module):
    """Sable encoder block."""

    embed_dim: int
    n_head: int
    n_agents: int
    net_config: DictConfig
    decay_scaling_factor: float

    def setup(self) -> None:
        # Initialize the RMSNorm layer normalization
        self.ln1 = nn.RMSNorm()
        self.ln2 = nn.RMSNorm()

        # Initialize the MultiScaleRetention
        self.retn = MultiScaleRetention(
            embed_dim=self.embed_dim,
            n_head=self.n_head,
            n_agents=self.n_agents,
            full_self_retention=True,  # Full retention for the encoder
            net_config=self.net_config,
            decay_scaling_factor=self.decay_scaling_factor,
        )

        # Initialize SwiGLU feedforward network
        self.ffn = SwiGLU(self.embed_dim, self.embed_dim)

    def __call__(
        self, x: chex.Array, hstate: chex.Array, dones: chex.Array, timestep_id: chex.Array
    ) -> chex.Array:
        """Applies Chunkwise MultiScaleRetention."""
        ret, updated_hstate = self.retn(
            key=x, query=x, value=x, hstate=hstate, dones=dones, timestep_id=timestep_id
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate

    def recurrent(self, x: chex.Array, hstate: chex.Array, timestep_id: chex.Array) -> chex.Array:
        """Applies Recurrent MultiScaleRetention."""
        ret, updated_hstate = self.retn.recurrent(
            key_n=x, query_n=x, value_n=x, hstate=hstate, timestep_id=timestep_id
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate


class Encoder(nn.Module):
    """Multi-block encoder consisting of multiple `EncoderBlock` modules."""

    n_block: int
    embed_dim: int
    n_head: int
    n_agents: int
    net_config: DictConfig
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
                self.n_agents,
                self.net_config,
                self.decay_scaling_factor,
                name=f"encoder_block_{block_id}",
            )
            for block_id in range(self.n_block)
        ]

    def __call__(
        self, obs: chex.Array, hstate: chex.Array, dones: chex.Array, timestep_id: chex.Array
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
            obs_rep, hs_new = block(self.ln(obs_rep), hs, dones, timestep_id)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)

        # Compute the value function
        v_loc = self.head(obs_rep)

        return v_loc, obs_rep, updated_hstate

    def recurrent(
        self, obs: chex.Array, hstate: chex.Array, timestep_id: chex.Array
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
            obs_rep, hs_new = block.recurrent(self.ln(obs_rep), hs, timestep_id)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)

        # Compute the value function
        v_loc = self.head(obs_rep)

        return v_loc, obs_rep, updated_hstate


class DecodeBlock(nn.Module):
    """Sable decoder block."""

    embed_dim: int
    n_head: int
    n_agents: int
    net_config: DictConfig
    decay_scaling_factor: float

    def setup(self) -> None:
        # Initialize the RMSNorm layer normalization
        self.ln1, self.ln2, self.ln3 = nn.RMSNorm(), nn.RMSNorm(), nn.RMSNorm()

        # Initialize the MultiScaleRetention
        self.retn1 = MultiScaleRetention(
            embed_dim=self.embed_dim,
            n_head=self.n_head,
            n_agents=self.n_agents,
            full_self_retention=False,  # Masked retention for the decoder
            net_config=self.net_config,
            decay_scaling_factor=self.decay_scaling_factor,
        )
        self.retn2 = MultiScaleRetention(
            embed_dim=self.embed_dim,
            n_head=self.n_head,
            n_agents=self.n_agents,
            full_self_retention=False,  # Masked retention for the decoder
            net_config=self.net_config,
            decay_scaling_factor=self.decay_scaling_factor,
        )

        # Initialize SwiGLU feedforward network
        self.ffn = SwiGLU(self.embed_dim, self.embed_dim)

    def __call__(
        self,
        x: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        dones: chex.Array,
        timestep_id: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Chunkwise MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1(
            key=x, query=x, value=x, hstate=hs1, dones=dones, timestep_id=timestep_id
        )
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2(
            key=ret,
            query=obs_rep,
            value=ret,
            hstate=hs2,
            dones=dones,
            timestep_id=timestep_id,
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)

    def recurrent(
        self,
        x: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        timestep_id: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Recurrent MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1.recurrent(
            key_n=x, query_n=x, value_n=x, hstate=hs1, timestep_id=timestep_id
        )
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2.recurrent(
            key_n=ret, query_n=obs_rep, value_n=ret, hstate=hs2, timestep_id=timestep_id
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)


class Decoder(nn.Module):
    """Multi-block decoder consisting of multiple `DecoderBlock` modules."""

    n_block: int
    embed_dim: int
    n_head: int
    n_agents: int
    action_dim: int
    net_config: DictConfig
    decay_scaling_factor: float = 1.0
    action_space_type: str = "discrete"

    def setup(self) -> None:
        # Initialize the RMSNorm layer normalization
        self.ln = nn.RMSNorm()

        # Initialize action encoder based on action space type
        if self.action_space_type == "discrete":
            self.action_encoder = nn.Sequential(
                [
                    nn.Dense(self.embed_dim, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                ],
            )
            # Set log_std to None for discrete action spaces as it is not used
            self.log_std = None
        else:
            self.action_encoder = nn.Sequential(
                [nn.Dense(self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
            )
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        # Initialize the head layer for the action logits
        self.head = nn.Sequential(
            [
                nn.Dense(self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
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
                self.n_agents,
                self.net_config,
                self.decay_scaling_factor,
                name=f"decoder_block_{block_id}",
            )
            for block_id in range(self.n_block)
        ]

    def __call__(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        dones: chex.Array,
        timestep_id: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Apply chunkwise decoding."""
        # Initialize the updated hidden states
        updated_hstates = (jnp.zeros_like(hstates[0]), jnp.zeros_like(hstates[1]))
        # Encode the action
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = jax.tree.map(lambda x, j=i: x[:, :, j], hstates)
            x, hs_new = block(
                x=x, obs_rep=obs_rep, hstates=hs, dones=dones, timestep_id=timestep_id
            )
            updated_hstates = jax.tree.map(
                lambda x, y, j=i: x.at[:, :, j].set(y), updated_hstates, hs_new
            )

        # Compute the action logits
        logit = self.head(x)

        return logit, updated_hstates

    def recurrent(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        timestep_id: chex.Array,
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
            x, hs_new = block.recurrent(x=x, obs_rep=obs_rep, hstates=hs, timestep_id=timestep_id)
            updated_hstates = jax.tree.map(
                lambda x, y, j=i: x.at[:, :, j].set(y), updated_hstates, hs_new
            )

        # Compute the action logits
        logit = self.head(x)

        return logit, updated_hstates


class SableNetwork(nn.Module):
    """Sable network module."""

    n_block: int
    embed_dim: int
    n_head: int
    n_agents: int
    action_dim: int
    net_config: DictConfig
    decay_scaling_factor: float = 1.0
    action_space_type: str = "discrete"

    def setup(self) -> None:
        # Check if the action space type is valid
        assert self.action_space_type in [
            "discrete",
            "continuous",
        ], "Invalid action space type"

        self.n_agents_per_chunk = self.n_agents
        if self.net_config.use_chunkwise:
            if self.net_config.type == "ff_sable":
                self.net_config.chunk_size = self.net_config.agents_chunk_size
                assert (
                    self.n_agents % self.net_config.chunk_size == 0
                ), "Number of agents should be divisible by chunk size"
                self.n_agents_per_chunk = self.net_config.chunk_size
            else:
                self.net_config.chunk_size = self.net_config.timestep_chunk_size * self.n_agents

        self.encoder = Encoder(
            self.n_block,
            self.embed_dim,
            self.n_head,
            self.n_agents_per_chunk,
            self.net_config,
            self.decay_scaling_factor,
        )
        self.decoder = Decoder(
            self.n_block,
            self.embed_dim,
            self.n_head,
            self.n_agents_per_chunk,
            self.action_dim,
            self.net_config,
            self.decay_scaling_factor,
            self.action_space_type,
        )

        # Set the executor and trainer functions
        (
            self.train_encoder_fn,
            self.train_decoder_fn,
            self.execute_encoder_fn,
            self.autoregressive_act,
        ) = self.setup_executor_trainer_fn()

        # Decay kappa for each head
        assert (
            self.decay_scaling_factor >= 0 and self.decay_scaling_factor <= 1
        ), "Decay scaling factor should be between 0 and 1"
        self.decay_kappas = 1 - jnp.exp(
            jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), self.n_head)
        )
        self.decay_kappas = self.decay_kappas * self.decay_scaling_factor

    def __call__(
        self,
        obs_carry: Observation,
        action: chex.Array,
        hstates: HiddenStates,
        dones: chex.Array,
        rng_key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Training phase."""
        # Get the observation, legal actions, and timestep id
        obs, legal_actions, timestep_id = (
            obs_carry.agents_view,
            obs_carry.action_mask,
            obs_carry.step_count,
        )
        # Apply the encoder
        v_loc, obs_rep, _ = self.train_encoder_fn(
            encoder=self.encoder, obs=obs, hstate=hstates[0], dones=dones, timestep_id=timestep_id
        )

        # Apply the decoder
        action_log, entropy = self.train_decoder_fn(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            legal_actions=legal_actions,
            hstates=hstates[1],
            dones=dones,
            timestep_id=timestep_id,
            rng_key=rng_key,
        )

        return v_loc, action_log, entropy

    def get_actions(
        self,
        obs_carry: Observation,
        hstates: HiddenStates,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, HiddenStates]:
        """Inference phase."""
        # Get the observation, legal actions, and timestep id
        obs, legal_actions, timestep_id = (
            obs_carry.agents_view,
            obs_carry.action_mask,
            obs_carry.step_count,
        )
        # Decay the hidden states
        decayed_hstates = jax.tree.map(
            lambda x: x * self.decay_kappas[None, :, None, None, None], hstates
        )

        # Apply the encoder
        v_loc, obs_rep, updated_enc_hs = self.execute_encoder_fn(
            encoder=self.encoder,
            obs=obs,
            decayed_hstate=decayed_hstates[0],
            timestep_id=timestep_id,
        )

        # Apply the decoder
        output_actions, output_actions_log, updated_dec_hs = self.autoregressive_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            legal_actions=legal_actions,
            hstates=decayed_hstates[1],
            timestep_id=timestep_id,
            key=key,
        )

        # Pack the hidden states
        updated_hs = HiddenStates(encoder_hstate=updated_enc_hs, decoder_hstate=updated_dec_hs)
        return output_actions, output_actions_log, v_loc, updated_hs

    def init_net(
        self,
        obs_carry: Observation,
        hstates: HiddenStates,
        key: chex.PRNGKey,
    ) -> Any:
        """Initializating the network."""

        return init_sable(  # noqa
            encoder=self.encoder,
            decoder=self.decoder,
            obs_carry=obs_carry,
            hstates=hstates,
            key=key,
        )

    def setup_executor_trainer_fn(self) -> Tuple:
        """Setup the executor and trainer functions."""

        # Set the executing encoder function based on the chunkwise setting.
        if self.net_config.use_chunkwise:
            # Define the trainer encoder in chunkwise setting.
            train_enc_fn = partial(train_encoder_chunkwise, chunk_size=self.net_config.chunk_size)  # noqa
            # Define the trainer decoder in chunkwise setting.
            act_fn = partial(act_chunkwise, chunk_size=self.net_config.chunk_size)  # noqa
            train_dec_fn = partial(train_decoder_fn, act_fn=act_fn, n_agents=self.n_agents)  # noqa
            # Define the executor encoder in chunkwise setting.
            if self.net_config.type == "ff_sable":
                execute_enc_fn = partial(
                    execute_encoder_chunkwise,  # noqa
                    chunk_size=self.net_config.chunk_size,
                )
            else:
                execute_enc_fn = partial(execute_encoder_parallel)  # noqa
        else:
            # Define the trainer encode when dealing with full sequence setting.
            train_enc_fn = partial(train_encoder_parallel)  # noqa
            # Define the trainer decoder when dealing with full sequence setting.
            train_dec_fn = partial(train_decoder_fn, act_fn=act_parallel, n_agents=self.n_agents)  # noqa
            # Define the executor encoder when dealing with full sequence setting.
            execute_enc_fn = partial(execute_encoder_parallel)  # noqa

        return train_enc_fn, train_dec_fn, execute_enc_fn, autoregressive_act  # noqa
