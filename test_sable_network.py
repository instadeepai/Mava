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

# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omegaconf import DictConfig

from mava.networks.sable_network import SableNetwork
from mava.systems.sable.types import HiddenStates
from mava.types import Observation

base_seed = 2
num_time_steps = 512
bsz = 16
num_agents = 4
obs_dim = 11

seq_len = num_agents * num_time_steps

retnet_embed_dim = 32
retnet_num_heads = 2
retnet_num_blocks = 2
num_chunks = 8

act_dim = 5

memory_config = DictConfig(
    {
        "type": "rec_sable",
        "decay_scaling_factor": 1.0,
        "timestep_positional_encoding": True,
        "timestep_chunk_size": num_time_steps // num_chunks,
        "chunk_size": num_agents * num_time_steps // num_chunks,
    }
)

net_config = DictConfig(
    {
        "n_block": retnet_num_blocks,
        "embed_dim": retnet_embed_dim,
        "n_head": retnet_num_heads,
    }
)

network = SableNetwork(
    n_agents=num_agents,
    n_agents_per_chunk=num_agents,
    action_dim=act_dim,
    net_config=net_config,
    memory_config=memory_config,
    action_space_type="discrete",
)

key = jax.random.PRNGKey(base_seed)
key, subkey = jax.random.split(key)

obs = jax.random.normal(subkey, (bsz, seq_len, obs_dim))
action_mask = jnp.ones((bsz, seq_len, act_dim), dtype=bool)

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

observation = Observation(
    agents_view=obs,
    action_mask=action_mask,
    step_count=step_counts,
)
init_hstates = HiddenStates(
    encoder=copy.deepcopy(init_hstate),
    decoder_self_retn=copy.deepcopy(init_hstate),
    decoder_cross_retn=copy.deepcopy(init_hstate),
)

params = network.init(
    init_key,
    observation=jax.tree.map(lambda x: x[0:1, 0:num_agents, ...], observation),
    hstates=jax.tree.map(lambda x: x[0:1, ...], init_hstates),
    key=init_key,
    method="get_actions",
)

JIT_FUNCTIONS = True

if JIT_FUNCTIONS:
    inf_apply = jax.jit(partial(network.apply, method="get_actions"))
    train_apply = jax.jit(network.apply)
else:
    inf_apply = partial(network.apply, method="get_actions")
    train_apply = network.apply

inference_actions = []
inference_log_probs = []
inference_values = []

hstates = copy.deepcopy(init_hstates)

for step in range(num_time_steps):
    key, step_key = jax.random.split(key)
    obs_i = jax.tree.map(
        lambda x, step=step: x[:, step * num_agents : (step + 1) * num_agents, ...], observation
    )
    action_i, log_prob_i, value_i, hstates = inf_apply(
        params,
        obs_i,
        hstates,
        step_key,
    )
    inference_actions.append(action_i)
    inference_log_probs.append(log_prob_i)
    inference_values.append(value_i)

inference_actions = jnp.concatenate(inference_actions, axis=1)
inference_log_probs = jnp.concatenate(inference_log_probs, axis=1)
inference_values = jnp.concatenate(inference_values, axis=1)

print(f"Inference actions shape: {inference_actions.shape}")
print(f"Inference log probs shape: {inference_log_probs.shape}")
print(f"Inference values shape: {inference_values.shape}")

hstates = copy.deepcopy(init_hstates)

train_value, train_log_prob, train_entropy = train_apply(
    params,
    observation,
    inference_actions,
    hstates,
    dones,
    key,
)

print(f"Train value shape: {train_value.shape}")
print(f"Train log prob shape: {train_log_prob.shape}")
print(f"Train entropy shape: {train_entropy.shape}")

print()
print(f"Value mean absolute error: {jnp.mean(jnp.abs(train_value - inference_values))}")
print(f"Value max absolute error: {jnp.max(jnp.abs(train_value - inference_values))}")
print(f"Log prob mean absolute error: {jnp.mean(jnp.abs(train_log_prob - inference_log_probs))}")
print(f"Log prob max absolute error: {jnp.max(jnp.abs(train_log_prob - inference_log_probs))}")
