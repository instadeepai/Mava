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
from functools import partial

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from mava.networks.retention import MultiScaleRetention

# jax.config.update("jax_enable_x64", True)

bsz = 4
num_agents = 4
obs_dim = 11
num_time_steps = 128
seq_len = num_agents * num_time_steps

retnet_embed_dim = 128
retnet_num_heads = 2
num_chunks = 1

memory_config = DictConfig(
    {
        "type": "rec_sable",
        "decay_scaling_factor": 1.0,
        "timestep_positional_encoding": True,
        "timestep_chunk_size": None,
    }
)

decay_kappas = 1 - jnp.exp(jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), retnet_num_heads))
decay_kappas *= memory_config.decay_scaling_factor
decay_kappas = jnp.log(decay_kappas)
decay_kappas = decay_kappas[None, :, None, None]

################################################################################
# Test unmasked MSR
################################################################################
msr_enc = MultiScaleRetention(
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

init_scale = jnp.ones((bsz, retnet_num_heads, 1, 1))

key, init_key = jax.random.split(key)

msr_enc_params = msr_enc.init(
    init_key,
    obs[0:1, 0:1, ...],
    obs[0:1, 0:1, ...],
    obs[0:1, 0:1, ...],
    init_hstate[0:1, ...],
    step_counts[0:1, 0:1],
    init_scale[0:1, ...],
    method="recurrent",
)
enc_jit_apply = partial(jax.jit(msr_enc.apply, static_argnames="num_chunks"))
enc_jit_inf = jax.jit(partial(msr_enc.apply, method="recurrent"))

hstate = copy.deepcopy(init_hstate)
scale = copy.deepcopy(init_scale)
act_output = []

# for the decoder we use the chunkwise
for step in range(num_time_steps):
    # todo: reset later
    new_scale = scale * jnp.exp(decay_kappas) + 1.0
    hstate_scale_factor = jnp.sqrt(scale) * jnp.exp(decay_kappas) / jnp.sqrt(new_scale)
    hstate = hstate * hstate_scale_factor
    scale = new_scale
    obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
    dones_i = dones[:, step * num_agents : (step + 1) * num_agents]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

    out, hstate = enc_jit_inf(
        msr_enc_params,
        obs_i,
        obs_i,
        obs_i,
        hstate,
        step_counts_i,
        kv_scale=scale,
    )
    act_output.append(out)

print("Never done test:")
print("Encoder:")
act_output = jnp.concatenate(act_output, axis=1)
print(act_output.shape)

hstate = copy.deepcopy(init_hstate)
scale = copy.deepcopy(init_scale)
train_out, _, _ = enc_jit_apply(
    msr_enc_params,
    obs,
    obs,
    obs,
    hstate,
    dones,
    step_counts,
    num_chunks=num_chunks,
    kv_scale=scale,
    inference=False,
)
print(train_out.shape)

total_error = jnp.mean(jnp.abs(train_out - act_output))
print(total_error)

################################################################################
# Test masked MSR
################################################################################

msr_dec = MultiScaleRetention(
    embed_dim=retnet_embed_dim,
    n_head=retnet_num_heads,
    n_agents=num_agents,
    memory_config=memory_config,
    masked=True,
    decay_scaling_factor=memory_config.decay_scaling_factor,
)

msr_dec_params = msr_dec.init(
    init_key,
    obs[0:1, 0:1, ...],
    obs[0:1, 0:1, ...],
    obs[0:1, 0:1, ...],
    init_hstate[0:1, ...],
    step_counts[0:1, 0:1],
    init_scale[0:1, ...],
    method="recurrent",
)
dec_jit_apply = partial(jax.jit(msr_dec.apply, static_argnames="num_chunks"))
dec_jit_inf = jax.jit(partial(msr_dec.apply, method="recurrent"))

hstate = copy.deepcopy(init_hstate)
scale = copy.deepcopy(init_scale)
act_output = []

# for the decoder we use the recurrent form at inference
for step in range(num_time_steps):
    obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

    new_scale = scale * jnp.exp(decay_kappas) + 1.0
    hstate_scale_factor = jnp.sqrt(scale) * jnp.exp(decay_kappas) / jnp.sqrt(new_scale)
    hstate = hstate * hstate_scale_factor
    scale = new_scale
    reset_done = dones[:, step * num_agents, None, None, None]
    hstate = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.zeros_like(x), x), hstate
    )
    timestep_outputs = []
    for agent in range(num_agents):
        obs_i_agent = obs_i[:, agent : agent + 1, ...]
        step_counts_i_agent = step_counts_i[:, agent : agent + 1, ...]

        out, hstate = dec_jit_inf(
            msr_dec_params,
            obs_i_agent,
            obs_i_agent,
            obs_i_agent,
            hstate,
            step_counts_i_agent,
            kv_scale=scale,
        )
        timestep_outputs.append(out)

    timestep_output = jnp.concatenate(timestep_outputs, axis=1)
    act_output.append(timestep_output)

print()
print("Decoder:")
act_output = jnp.concatenate(act_output, axis=1)
print(act_output.shape)

hstate = copy.deepcopy(init_hstate)
scale = copy.deepcopy(init_scale)
train_out, _, _ = dec_jit_apply(
    msr_dec_params,
    obs,
    obs,
    obs,
    hstate,
    dones,
    step_counts,
    num_chunks=num_chunks,
    kv_scale=scale,
    inference=False,
)
print(train_out.shape)

total_error = jnp.mean(jnp.abs(train_out - act_output))
print(total_error)

print()
print("With done test:")

key, done_key = jax.random.split(key)
dones = jnp.repeat(  # dones are the same per agent so repeat them
    jax.random.randint(done_key, (bsz, num_time_steps), 0, 2).astype(bool), num_agents, axis=1
)

print("Encoder:")
hstate = copy.deepcopy(init_hstate)
scale = copy.deepcopy(init_scale)

act_output = []
for step in range(num_time_steps):
    new_scale = scale * jnp.exp(decay_kappas) + 1.0
    hstate_scale_factor = jnp.sqrt(scale) * jnp.exp(decay_kappas) / jnp.sqrt(new_scale)
    hstate = hstate * hstate_scale_factor
    scale = new_scale
    reset_done = dones[:, step * num_agents, None, None, None]
    hstate = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.zeros_like(x), x), hstate
    )
    scale = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.ones_like(x), x), scale
    )
    obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
    dones_i = dones[:, step : step + 1]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

    out, hstate = enc_jit_inf(
        msr_enc_params,
        obs_i,
        obs_i,
        obs_i,
        hstate,
        step_counts_i,
        kv_scale=scale,
    )
    act_output.append(out)

act_output = jnp.concatenate(act_output, axis=1)
print(act_output.shape)

hstate = copy.deepcopy(init_hstate)
scale = copy.deepcopy(init_scale)
train_out, _, _ = enc_jit_apply(
    msr_enc_params,
    obs,
    obs,
    obs,
    hstate,
    dones,
    step_counts,
    num_chunks=1,
    kv_scale=scale,
    inference=False,
)
print(train_out.shape)

total_error = jnp.mean(jnp.abs(train_out - act_output))
print(total_error)

print()
print("Decoder:")
hstate = copy.deepcopy(init_hstate)
act_output = []
for step in range(num_time_steps):
    obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

    new_scale = scale * jnp.exp(decay_kappas) + 1.0
    hstate_scale_factor = jnp.sqrt(scale) * jnp.exp(decay_kappas) / jnp.sqrt(new_scale)
    hstate = hstate * hstate_scale_factor
    scale = new_scale
    reset_done = dones[:, step * num_agents, None, None, None]
    hstate = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.zeros_like(x), x), hstate
    )
    scale = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.ones_like(x), x), scale
    )

    timestep_outputs = []
    for agent in range(num_agents):
        obs_i_agent = obs_i[:, agent : agent + 1, ...]
        step_counts_i_agent = step_counts_i[:, agent : agent + 1, ...]

        out, hstate = dec_jit_inf(
            msr_dec_params,
            obs_i_agent,
            obs_i_agent,
            obs_i_agent,
            hstate,
            step_counts_i_agent,
            kv_scale=scale,
        )
        timestep_outputs.append(out)

    timestep_output = jnp.concatenate(timestep_outputs, axis=1)
    act_output.append(timestep_output)

act_output = jnp.concatenate(act_output, axis=1)
print(act_output.shape)

hstate = copy.deepcopy(init_hstate)
scale = copy.deepcopy(init_scale)
train_out, _, _ = dec_jit_apply(
    msr_dec_params,
    obs,
    obs,
    obs,
    hstate,
    dones,
    step_counts,
    num_chunks=num_chunks,
    kv_scale=scale,
    inference=False,
)

print(train_out.shape)

total_error = jnp.mean(jnp.abs(train_out - act_output))
print(total_error)
