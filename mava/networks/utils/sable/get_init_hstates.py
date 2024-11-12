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

import jax.numpy as jnp

from mava.systems.sable.types import HiddenStates, SableNetworkConfig


def get_init_hidden_state(actor_net_config: SableNetworkConfig, batch_size: int) -> HiddenStates:
    """Initializes the hidden states for the encoder and decoder."""
    # Compute the hidden state size based on embedding dimension and number of heads
    hidden_size = actor_net_config.embed_dim // actor_net_config.n_head

    # Define the shape of the hidden states
    hidden_state_shape = (
        batch_size,
        actor_net_config.n_head,
        actor_net_config.n_block,
        hidden_size,
        hidden_size,
    )

    # Initialize hidden states for encoder and decoder
    dec_hs_self_retn = jnp.zeros(hidden_state_shape)
    dec_hs_cross_retn = jnp.zeros(hidden_state_shape)
    enc_hs = jnp.zeros(hidden_state_shape)
    hidden_states = HiddenStates(
        encoder=enc_hs,
        decoder_self_retn=dec_hs_self_retn,
        decoder_cross_retn=dec_hs_cross_retn,
    )
    return hidden_states
