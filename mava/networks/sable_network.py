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
from typing import Optional, Tuple

import chex
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal
from jax import tree
from omegaconf import DictConfig

from mava.networks.retention import MultiScaleRetention
from mava.networks.torsos import SwiGLU
from mava.networks.utils.sable import (
    act_encoder_fn,
    continuous_autoregressive_act,
    continuous_train_decoder_fn,
    discrete_autoregressive_act,
    discrete_train_decoder_fn,
    train_encoder_fn,
)
from mava.systems.sable.types import HiddenStates, SableNetworkConfig
from mava.types import Observation
from mava.utils.network_utils import _CONTINUOUS, _DISCRETE


class EncodeBlock(nn.Module):
    """Sable encoder block."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    n_agents: int

    def setup(self) -> None:
        self.ln1 = nn.RMSNorm()
        self.ln2 = nn.RMSNorm()

        self.retn = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            n_agents=self.n_agents,
            masked=False,  # Full retention for the encoder
            memory_config=self.memory_config,
            decay_scaling_factor=self.memory_config.decay_scaling_factor,
        )

        self.ffn = SwiGLU(self.net_config.embed_dim, self.net_config.embed_dim)

    def __call__(
        self, x: chex.Array, hstate: chex.Array, dones: chex.Array, step_count: chex.Array
    ) -> chex.Array:
        """Applies Chunkwise MultiScaleRetention."""
        ret, updated_hstate = self.retn(
            key=x, query=x, value=x, hstate=hstate, dones=dones, step_count=step_count
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate

    def recurrent(self, x: chex.Array, hstate: chex.Array, step_count: chex.Array) -> chex.Array:
        """Applies Recurrent MultiScaleRetention."""
        ret, updated_hstate = self.retn.recurrent(
            key_n=x, query_n=x, value_n=x, hstate=hstate, step_count=step_count
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate


class Encoder(nn.Module):
    """Multi-block encoder consisting of multiple `EncoderBlock` modules."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    n_agents: int

    def setup(self) -> None:
        self.ln = nn.RMSNorm()

        self.obs_encoder = nn.Sequential(
            [
                nn.RMSNorm(),
                nn.Dense(
                    self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2)), use_bias=False
                ),
                nn.gelu,
            ],
        )
        self.head = nn.Sequential(
            [
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.RMSNorm(),
                nn.Dense(1, kernel_init=orthogonal(0.01)),
            ],
        )

        self.blocks = [
            EncodeBlock(
                self.net_config,
                self.memory_config,
                self.n_agents,
                name=f"encoder_block_{block_id}",
            )
            for block_id in range(self.net_config.n_block)
        ]

    def __call__(
        self, obs: chex.Array, hstate: chex.Array, dones: chex.Array, step_count: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply chunkwise encoding."""
        updated_hstate = jnp.zeros_like(hstate)
        obs_rep = self.obs_encoder(obs)

        # Apply the encoder blocks
        for i, block in enumerate(self.blocks):
            hs = hstate[:, :, i]  # Get the hidden state for the current block
            # Apply the chunkwise encoder block
            obs_rep, hs_new = block(self.ln(obs_rep), hs, dones, step_count)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)

        value = self.head(obs_rep)

        return value, obs_rep, updated_hstate

    def recurrent(
        self, obs: chex.Array, hstate: chex.Array, step_count: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply recurrent encoding."""
        updated_hstate = jnp.zeros_like(hstate)
        obs_rep = self.obs_encoder(obs)

        # Apply the encoder blocks
        for i, block in enumerate(self.blocks):
            hs = hstate[:, :, i]  # Get the hidden state for the current block
            # Apply the recurrent encoder block
            obs_rep, hs_new = block.recurrent(self.ln(obs_rep), hs, step_count)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)

        # Compute the value function
        value = self.head(obs_rep)

        return value, obs_rep, updated_hstate


class DecodeBlock(nn.Module):
    """Sable decoder block."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    n_agents: int

    def setup(self) -> None:
        self.ln1, self.ln2, self.ln3 = nn.RMSNorm(), nn.RMSNorm(), nn.RMSNorm()

        self.retn1 = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            n_agents=self.n_agents,
            masked=True,  # Masked retention for the decoder
            memory_config=self.memory_config,
            decay_scaling_factor=self.memory_config.decay_scaling_factor,
        )
        self.retn2 = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            n_agents=self.n_agents,
            masked=True,  # Masked retention for the decoder
            memory_config=self.memory_config,
            decay_scaling_factor=self.memory_config.decay_scaling_factor,
        )

        self.ffn = SwiGLU(self.net_config.embed_dim, self.net_config.embed_dim)

    def __call__(
        self,
        x: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        dones: chex.Array,
        step_count: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Chunkwise MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1(
            key=x, query=x, value=x, hstate=hs1, dones=dones, step_count=step_count
        )
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2(
            key=ret,
            query=obs_rep,
            value=ret,
            hstate=hs2,
            dones=dones,
            step_count=step_count,
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)

    def recurrent(
        self,
        x: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        step_count: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Recurrent MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1.recurrent(
            key_n=x, query_n=x, value_n=x, hstate=hs1, step_count=step_count
        )
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2.recurrent(
            key_n=ret, query_n=obs_rep, value_n=ret, hstate=hs2, step_count=step_count
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)


class Decoder(nn.Module):
    """Multi-block decoder consisting of multiple `DecoderBlock` modules."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    n_agents: int
    action_dim: int
    action_space_type: str = _DISCRETE

    def setup(self) -> None:
        self.ln = nn.RMSNorm()

        use_bias = self.action_space_type == _CONTINUOUS
        self.action_encoder = nn.Sequential(
            [
                nn.Dense(
                    self.net_config.embed_dim,
                    use_bias=use_bias,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                ),
                nn.gelu,
            ],
        )

        # Always initialize log_std but set to None for discrete action spaces
        # This ensures the attribute exists but signals it should not be used.
        self.log_std = (
            self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            if self.action_space_type == _CONTINUOUS
            else None
        )

        self.head = nn.Sequential(
            [
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.RMSNorm(),
                nn.Dense(self.action_dim, kernel_init=orthogonal(0.01)),
            ],
        )

        self.blocks = [
            DecodeBlock(
                self.net_config,
                self.memory_config,
                self.n_agents,
                name=f"decoder_block_{block_id}",
            )
            for block_id in range(self.net_config.n_block)
        ]

    def __call__(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        dones: chex.Array,
        step_count: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Apply chunkwise decoding."""
        updated_hstates = tree.map(jnp.zeros_like, hstates)
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = tree.map(lambda x, j=i: x[:, :, j], hstates)
            x, hs_new = block(x=x, obs_rep=obs_rep, hstates=hs, dones=dones, step_count=step_count)
            updated_hstates = tree.map(
                lambda x, y, j=i: x.at[:, :, j].set(y), updated_hstates, hs_new
            )

        logit = self.head(x)

        return logit, updated_hstates

    def recurrent(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        step_count: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Apply recurrent decoding."""
        updated_hstates = tree.map(jnp.zeros_like, hstates)
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = tree.map(lambda x, i=i: x[:, :, i], hstates)
            x, hs_new = block.recurrent(x=x, obs_rep=obs_rep, hstates=hs, step_count=step_count)
            updated_hstates = tree.map(
                lambda x, y, j=i: x.at[:, :, j].set(y), updated_hstates, hs_new
            )

        logit = self.head(x)

        return logit, updated_hstates


class SableNetwork(nn.Module):
    """Sable network module."""

    n_agents: int
    n_agents_per_chunk: int
    action_dim: int
    net_config: SableNetworkConfig
    memory_config: DictConfig
    action_space_type: str = _DISCRETE

    def setup(self) -> None:
        if self.action_space_type not in [_DISCRETE, _CONTINUOUS]:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

        assert (
            self.memory_config.decay_scaling_factor >= 0
            and self.memory_config.decay_scaling_factor <= 1
        ), "Decay scaling factor should be between 0 and 1"

        # Decay kappa for each head
        self.decay_kappas = 1 - jnp.exp(
            jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), self.net_config.n_head)
        )
        self.decay_kappas = self.decay_kappas * self.memory_config.decay_scaling_factor
        self.decay_kappas = self.decay_kappas[None, :, None, None, None]

        self.encoder = Encoder(
            self.net_config,
            self.memory_config,
            self.n_agents_per_chunk,
        )
        self.decoder = Decoder(
            self.net_config,
            self.memory_config,
            self.n_agents_per_chunk,
            self.action_dim,
            self.action_space_type,
        )

        # Set the actor and trainer functions
        self.train_encoder_fn = partial(
            train_encoder_fn,
            chunk_size=self.memory_config.chunk_size,
        )
        self.act_encoder_fn = partial(
            act_encoder_fn,
            chunk_size=self.n_agents_per_chunk,
        )
        if self.action_space_type == _CONTINUOUS:
            self.train_decoder_fn = partial(
                continuous_train_decoder_fn,
                n_agents=self.n_agents,
                chunk_size=self.memory_config.chunk_size,
                action_dim=self.action_dim,
            )
            self.autoregressive_act = partial(
                continuous_autoregressive_act, action_dim=self.action_dim
            )
        else:
            self.train_decoder_fn = partial(
                discrete_train_decoder_fn,
                n_agents=self.n_agents,
                chunk_size=self.memory_config.chunk_size,
            )
            self.autoregressive_act = discrete_autoregressive_act  # type: ignore

    def __call__(
        self,
        observation: Observation,
        action: chex.Array,
        hstates: HiddenStates,
        dones: chex.Array,
        rng_key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Training phase."""
        obs, legal_actions, step_count = (
            observation.agents_view,
            observation.action_mask,
            observation.step_count,
        )
        value, obs_rep, _ = self.train_encoder_fn(
            encoder=self.encoder, obs=obs, hstate=hstates[0], dones=dones, step_count=step_count
        )

        action_log, entropy = self.train_decoder_fn(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            legal_actions=legal_actions,
            hstates=hstates[1:],
            dones=dones,
            step_count=step_count,
            rng_key=rng_key,
        )

        value = jnp.squeeze(value, axis=-1)
        return value, action_log, entropy

    def get_actions(
        self,
        observation: Observation,
        hstates: HiddenStates,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, HiddenStates]:
        """Inference phase."""
        obs, legal_actions, step_count = (
            observation.agents_view,
            observation.action_mask,
            observation.step_count,
        )

        # Decay the hidden states: each timestep we decay the hidden states once
        decayed_hstates = tree.map(lambda x: x * self.decay_kappas, hstates)

        value, obs_rep, updated_enc_hs = self.act_encoder_fn(
            encoder=self.encoder,
            obs=obs,
            decayed_hstate=decayed_hstates[0],
            step_count=step_count,
        )

        output_actions, output_actions_log, updated_dec_hs = self.autoregressive_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            legal_actions=legal_actions,
            hstates=decayed_hstates[1:],
            step_count=step_count,
            key=key,
        )

        updated_hs = HiddenStates(
            encoder=updated_enc_hs,
            decoder_self_retn=updated_dec_hs[0],
            decoder_cross_retn=updated_dec_hs[1],
        )

        value = jnp.squeeze(value, axis=-1)
        return output_actions, output_actions_log, value, updated_hs
