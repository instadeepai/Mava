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
)

hstate = copy.deepcopy(init_hstate)
act_output = []


# for the decoder we use the chunkwise
for step in range(num_time_steps):
    # todo: reset later
    hstate = hstate * decay_kappas
    obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
    dones_i = dones[:, step * num_agents : (step + 1) * num_agents]
    step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

    out, hstate = msr.apply(params, obs_i, obs_i, obs_i, hstate, step_counts_i, method="recurrent")
    act_output.append(out)

act_output = jnp.concatenate(act_output, axis=1)
print(f"Act output shape: {act_output.shape}")

hstate = copy.deepcopy(init_hstate)
train_out, _ = msr.apply(params, obs, obs, obs, hstate, dones, step_counts)

print(f"Train output shape: {train_out.shape}")

total_error = jnp.mean(jnp.abs(train_out - act_output))
print(f"Total error: {total_error}")

print(jnp.abs(train_out - act_output))
