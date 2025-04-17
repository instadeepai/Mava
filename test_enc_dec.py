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
# limitations under the License

import copy
from functools import partial

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from mava.networks.sable_network import Decoder, Encoder

base_seed = 2

bsz = 16
num_agents = 4
obs_dim = 11
num_time_steps = 4096
seq_len = num_agents * num_time_steps

retnet_embed_dim = 128
retnet_num_heads = 2
retnet_num_blocks = 2
num_chunks = 16

act_dim = 5

memory_config = DictConfig(
    {
        "type": "rec_sable",
        "decay_scaling_factor": 1.0,
        "timestep_positional_encoding": True,
        "timestep_chunk_size": None,
    }
)

net_config = DictConfig(
    {
        "n_block": retnet_num_blocks,
        "embed_dim": retnet_embed_dim,
        "n_head": retnet_num_heads,
    }
)
decay_kappas = 1 - jnp.exp(jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), retnet_num_heads))
decay_kappas *= memory_config.decay_scaling_factor
decay_kappas = jnp.log(decay_kappas)
decay_kappas = decay_kappas[None, :, None, None, None]

JIT_FUNCTIONS = True

key = jax.random.PRNGKey(base_seed)
key, subkey = jax.random.split(key)

obs = jax.random.normal(subkey, (bsz, seq_len, obs_dim))

# assuming no resets
dones = jnp.zeros((bsz, seq_len), dtype=bool)

init_hstate = jnp.zeros(
    (
        bsz,
        retnet_num_heads,
        retnet_num_blocks,
        retnet_embed_dim // retnet_num_heads,
        retnet_embed_dim // retnet_num_heads,
    )
)
step_counts = jnp.arange(num_time_steps)
step_counts = step_counts[None, ...].repeat(bsz, axis=0)[..., None].repeat(num_agents, axis=-1)
step_counts = step_counts.reshape(bsz, seq_len)

key, init_key = jax.random.split(key)

################################################################################
# Test Encoder
################################################################################

enc_network = Encoder(
    net_config=net_config,
    memory_config=memory_config,
    n_agents=num_agents,
)

enc_network_params = enc_network.init(
    init_key,
    obs[0:1, 0:num_agents, ...],
    init_hstate[0:1, ...],
    dones[0:1, 0:num_agents],
    step_counts[0:1, 0:num_agents],
    num_chunks=1,
)

if JIT_FUNCTIONS:
    enc_jit_apply = jax.jit(enc_network.apply, static_argnames=["num_chunks"])
    enc_jit_inf = jax.jit(partial(enc_network.apply, num_chunks=1))
else:
    enc_jit_apply = enc_network.apply
    enc_jit_inf = partial(enc_network.apply, num_chunks=1)

hstate = copy.deepcopy(init_hstate)

inference_encoded_obs = []
inference_value = []

for step in range(num_time_steps):
    obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]
    dones_i = dones[:, step * num_agents : (step + 1) * num_agents]

    reset_done = dones[:, step * num_agents, None, None, None, None]
    hstate = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.zeros_like(x), x), hstate
    )

    act_value, act_obs_rep, hstate = enc_jit_inf(
        enc_network_params,
        obs_i,
        hstate,
        jnp.zeros_like(dones_i, dtype=bool),
        step_counts_i,
    )

    inference_encoded_obs.append(act_obs_rep)
    inference_value.append(act_value)

inference_encoded_obs = jnp.concatenate(inference_encoded_obs, axis=1)
inference_value = jnp.concatenate(inference_value, axis=1)

print("Never done test:")
print("Encoder:")
print(inference_encoded_obs.shape)
print(inference_value.shape)

hstate = copy.deepcopy(init_hstate)

train_value_out, train_obs_rep_out, _ = enc_jit_apply(
    enc_network_params,
    obs,
    hstate,
    dones,
    step_counts,
    num_chunks=num_chunks,
)

print(train_obs_rep_out.shape)
print(train_value_out.shape)

total_value_error = jnp.mean(jnp.abs(train_value_out - inference_value))
print(f"Mean absolute value error: {total_value_error}")
print(f"Max absolute value error: {jnp.max(jnp.abs(train_value_out - inference_value))}")

total_obs_error = jnp.mean(jnp.abs(train_obs_rep_out - inference_encoded_obs))
print(f"Mean absolute encoded obs error: {total_obs_error}")
print(
    f"Max absolute encoded obs error: {jnp.max(jnp.abs(train_obs_rep_out - inference_encoded_obs))}"
)

################################################################################
# Test Decoder
################################################################################

dec_network = Decoder(
    net_config=net_config,
    memory_config=memory_config,
    n_agents=num_agents,
    action_dim=act_dim,
    action_space_type="discrete",
)

key, subkey = jax.random.split(key)
embedded_obs = jax.random.normal(subkey, (bsz, seq_len, retnet_embed_dim))

key, subkey = jax.random.split(key)
actions = jax.random.randint(subkey, (bsz, seq_len), 0, act_dim)
one_hot_actions = jax.nn.one_hot(actions, act_dim, dtype=float)

dec_network_params = dec_network.init(
    init_key,
    action=one_hot_actions[0:1, 0:1, ...],
    obs_rep=embedded_obs[0:1, 0:1, ...],
    hstates=(init_hstate[0:1, ...], init_hstate[0:1, ...]),
    step_count=step_counts[0:1, 0:1],
    method="recurrent",
)

if JIT_FUNCTIONS:
    dec_jit_apply = jax.jit(dec_network.apply, static_argnames=["num_chunks"])
    dec_jit_inf = jax.jit(partial(dec_network.apply, method="recurrent"))
else:
    dec_jit_apply = dec_network.apply
    dec_jit_inf = partial(dec_network.apply, method="recurrent")

hstate = (copy.deepcopy(init_hstate), copy.deepcopy(init_hstate))

act_logits = []

for step in range(num_time_steps):
    hstate = jax.tree.map(lambda x: x * jnp.exp(decay_kappas), hstate)

    obs_i = embedded_obs[:, step * num_agents : (step + 1) * num_agents, ...]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]
    actions_i = one_hot_actions[:, step * num_agents : (step + 1) * num_agents, ...]

    reset_done = dones[:, step * num_agents, None, None, None, None]
    hstate = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.zeros_like(x), x), hstate
    )
    timestep_outputs = []
    for agent in range(num_agents):
        obs_i_agent = obs_i[:, agent : agent + 1, ...]
        step_counts_i_agent = step_counts_i[:, agent : agent + 1, ...]
        actions_i_agent = actions_i[:, agent : agent + 1, ...]
        out, hstate = dec_jit_inf(
            dec_network_params,
            actions_i_agent,
            obs_i_agent,
            hstate,
            step_counts_i_agent,
        )
        timestep_outputs.append(out)

    act_logits.append(jnp.concatenate(timestep_outputs, axis=1))

act_logits = jnp.concatenate(act_logits, axis=1)

print("Never done test:")
print("Decoder:")
print(act_logits.shape)
hstate = (copy.deepcopy(init_hstate), copy.deepcopy(init_hstate))

train_logits, _ = dec_jit_apply(
    dec_network_params,
    one_hot_actions,
    embedded_obs,
    hstate,
    dones,
    step_counts,
    num_chunks=num_chunks,
)

print(train_logits.shape)
total_logits_error = jnp.mean(jnp.abs(train_logits - act_logits))
print(f"Mean absolute logits error: {total_logits_error}")
print(f"Max absolute logits error: {jnp.max(jnp.abs(train_logits - act_logits))}")

print()
print("With done test:")

key, done_key = jax.random.split(key)
dones = jnp.repeat(  # dones are the same per agent so repeat them
    jax.random.randint(done_key, (bsz, num_time_steps), 0, 2).astype(bool), num_agents, axis=1
)

print("Encoder:")
hstate = copy.deepcopy(init_hstate)

inference_encoded_obs = []
inference_value = []

for step in range(num_time_steps):
    reset_done = dones[:, step * num_agents, None, None, None, None]
    hstate = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.zeros_like(x), x), hstate
    )
    obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
    dones_i = dones[:, step * num_agents : (step + 1) * num_agents]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

    act_value, act_obs_rep, hstate = enc_jit_inf(
        enc_network_params,
        obs_i,
        hstate,
        jnp.zeros_like(dones_i, dtype=bool),
        step_counts_i,
    )

    inference_encoded_obs.append(act_obs_rep)
    inference_value.append(act_value)

inference_encoded_obs = jnp.concatenate(inference_encoded_obs, axis=1)
inference_value = jnp.concatenate(inference_value, axis=1)

print(inference_encoded_obs.shape)
print(inference_value.shape)

hstate = copy.deepcopy(init_hstate)

train_value_out, train_obs_rep_out, _ = enc_jit_apply(
    enc_network_params,
    obs,
    hstate,
    dones,
    step_counts,
    num_chunks=num_chunks,
)

print(train_obs_rep_out.shape)
print(train_value_out.shape)

total_value_error = jnp.mean(jnp.abs(train_value_out - inference_value))
print(f"Mean absolute value error: {total_value_error}")
print(f"Max absolute value error: {jnp.max(jnp.abs(train_value_out - inference_value))}")

total_obs_error = jnp.mean(jnp.abs(train_obs_rep_out - inference_encoded_obs))
print(f"Mean absolute encoded obs error: {total_obs_error}")
print(
    f"Max absolute encoded obs error: {jnp.max(jnp.abs(train_obs_rep_out - inference_encoded_obs))}"
)

print("Decoder:")
hstate = (copy.deepcopy(init_hstate), copy.deepcopy(init_hstate))

act_logits = []

for step in range(num_time_steps):
    hstate = jax.tree.map(lambda x: x * jnp.exp(decay_kappas), hstate)
    obs_i = embedded_obs[:, step * num_agents : (step + 1) * num_agents, ...]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]
    actions_i = one_hot_actions[:, step * num_agents : (step + 1) * num_agents, ...]

    reset_done = dones[:, step * num_agents, None, None, None, None]
    hstate = jax.tree.map(
        lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.zeros_like(x), x), hstate
    )

    timestep_outputs = []
    for agent in range(num_agents):
        obs_i_agent = obs_i[:, agent : agent + 1, ...]
        step_counts_i_agent = step_counts_i[:, agent : agent + 1, ...]
        actions_i_agent = actions_i[:, agent : agent + 1, ...]
        out, hstate = dec_jit_inf(
            dec_network_params,
            actions_i_agent,
            obs_i_agent,
            hstate,
            step_counts_i_agent,
        )
        timestep_outputs.append(out)

    act_logits.append(jnp.concatenate(timestep_outputs, axis=1))

act_logits = jnp.concatenate(act_logits, axis=1)

print(act_logits.shape)

hstate = (copy.deepcopy(init_hstate), copy.deepcopy(init_hstate))

train_logits, _ = dec_jit_apply(
    dec_network_params,
    one_hot_actions,
    embedded_obs,
    hstate,
    dones,
    step_counts,
    num_chunks=num_chunks,
)

total_logits_error = jnp.mean(jnp.abs(train_logits - act_logits))
print(f"Mean absolute logits error: {total_logits_error}")
print(f"Max absolute logits error: {jnp.max(jnp.abs(train_logits - act_logits))}")
