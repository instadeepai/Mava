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

import copy

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from mava.networks.retention import MultiScaleRetention

# jax.config.update("jax_enable_x64", True)

bsz = 16
num_agents = 4
obs_dim = 11
num_time_steps = 100
seq_len = num_agents * num_time_steps

retnet_embed_dim = 32
retnet_num_heads = 2
num_chunks = 1

memory_config = DictConfig(
    {
        "type": "rec_sable",
        "decay_scaling_factor": 0.3,
        "timestep_positional_encoding": True,
        "timestep_chunk_size": None,
    }
)

decay_kappas = 1 - jnp.exp(jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), retnet_num_heads))
decay_kappas *= memory_config.decay_scaling_factor
decay_kappas = jnp.log(decay_kappas)
decay_kappas = decay_kappas[None, :, None, None]

msr = MultiScaleRetention(
    embed_dim=retnet_embed_dim,
    n_head=retnet_num_heads,
    n_agents=num_agents,
    memory_config=memory_config,
    masked=False,
    decay_scaling_factor=memory_config.decay_scaling_factor,
)

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

obs = jax.random.normal(subkey, (bsz, seq_len, retnet_embed_dim))

# assuming no resets
dones = jnp.zeros((bsz, seq_len), dtype=bool)

init_hstate = jnp.zeros(
    (
        bsz,
        retnet_num_heads,
        retnet_embed_dim // retnet_num_heads,
        retnet_embed_dim // retnet_num_heads,
    )
)
step_counts = jnp.arange(num_time_steps)
step_counts = step_counts[None, ...].repeat(bsz, axis=0)[..., None].repeat(num_agents, axis=-1)
step_counts = step_counts.reshape(bsz, seq_len)

key, init_key = jax.random.split(key)
params = msr.init(
    init_key,
    obs,
    obs,
    obs,
    init_hstate,
    dones,
    step_counts,
    num_chunks=num_chunks,
)

hstate = copy.deepcopy(init_hstate)
act_output = []

# for the decoder we use the chunkwise
for step in range(num_time_steps):
    # todo: reset later
    hstate = hstate * jnp.exp(decay_kappas)
    obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
    dones_i = dones[:, step * num_agents : (step + 1) * num_agents]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

    out, hstate = msr.apply(
        params,
        obs_i,
        obs_i,
        obs_i,
        hstate,
        dones_i,
        step_counts_i,
        num_chunks=num_chunks,
        inference=True,
    )
    act_output.append(out)

act_output = jnp.concatenate(act_output, axis=1)
print(act_output.shape)

hstate = copy.deepcopy(init_hstate)
train_out, _ = msr.apply(
    params, obs, obs, obs, hstate, dones, step_counts, num_chunks=1, inference=False
)
print(train_out.shape)

total_error = jnp.mean(jnp.abs(train_out - act_output))
print(total_error)

# print(jnp.abs(train_out - act_output))
