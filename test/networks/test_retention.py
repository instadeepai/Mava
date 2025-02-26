from functools import partial

import jax
import jax.numpy as jnp
import pytest

from mava.networks.retention import MultiScaleRetention
from omegaconf import DictConfig

jax.config.update("jax_enable_x64", True)

TOL = 1e-14


@pytest.mark.parametrize("never_done", [True, False])
@pytest.mark.parametrize("begining_of_episode", [True])
@pytest.mark.parametrize("encoder", [True, False])
def test_chunkwise_and_recurrent(never_done, begining_of_episode, encoder):
    batch = 16

    n_agents = 4
    n_timesteps = 10
    seq_len = n_agents * n_timesteps

    num_heads = 4
    hidden_size = 32
    head_size = hidden_size // num_heads
    memory_config = DictConfig(
    {
        "type": "rec_sable",
        "decay_scaling_factor": 0.3,
        "timestep_positional_encoding": True,
        "timestep_chunk_size": None,
    }
)
    gammas = 1 - jnp.exp(jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), num_heads))
    gammas = jnp.log(gammas * memory_config.decay_scaling_factor)
    gammas = gammas[None, :, None, None]

    key = jax.random.PRNGKey(2454252)
    dkey, hkey, key = jax.random.split(key, 3)

    obs = jax.random.uniform(key, (batch, seq_len, hidden_size))

    if not never_done:  # if you want no dones
        dones = jnp.zeros((batch, seq_len), dtype=bool)
    else:  # if you want random dones
        dones = jnp.repeat(  # dones are the same per agent so repeat them
            jax.random.randint(dkey, (batch, n_timesteps), 0, 2).astype(bool), n_agents, axis=1
        )

    hidden_state_shape = (
        batch,
        num_heads,
        head_size,
        head_size,
    )
    scale_shape = (
        batch,
        num_heads,
        1,
        1,
    )
    

    if begining_of_episode:  # if you want to simulate starting at the first timestep
        init_hs = jnp.zeros(hidden_state_shape)
        init_scale = jnp.ones(scale_shape)

    else:  # if you want to simulate starting at a random timestep
        init_hs = jax.random.uniform(hkey, hidden_state_shape)
        init_scale = jnp.ones(scale_shape)

    ret = MultiScaleRetention(hidden_size, num_heads, n_agents, memory_config, encoder, memory_config.decay_scaling_factor)
    rec_apply = jax.jit(partial(ret.apply, method=ret.recurrent))

    params = ret.init(key, obs, obs, obs, dones)
    act_output = []

    act_hs = jnp.copy(init_hs)
    act_scale = jnp.copy(init_scale)
    for step in range(n_timesteps):
        updated_scale = jax.tree.map(lambda x: x * jnp.exp(gammas) + 1, act_scale)
        scale_factors = jax.tree.map(lambda x, y: jnp.sqrt(x) * jnp.exp(gammas) / jnp.sqrt(y), act_scale, updated_scale)
        h_leaves, h_treedef = jax.tree_util.tree_flatten(act_hs)
        s_leaves, _ = jax.tree_util.tree_flatten(scale_factors)
        decayed_leaves = [h * s for h, s in zip(h_leaves, s_leaves)]
        act_hs = jax.tree_util.tree_unflatten(h_treedef, decayed_leaves)
        if encoder:  # chunkwise encoder acting
            reset_done = dones[:, step * n_agents, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            act_hs = jax.tree_map(lambda x: jnp.where(reset_done, jnp.zeros_like(x), x), act_hs)

            obs_i = obs[:, step * n_agents : (step + 1) * n_agents]
            dones_i = dones[:, step * n_agents : (step + 1) * n_agents]
            x, act_hs = rec_apply(params, obs_i, obs_i, obs_i, act_hs, dones_i)
            act_output.append(x)
        else:  # recurrent decoder
            reset_done = dones[:, step * n_agents, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            act_hs = jax.tree_map(lambda x: jnp.where(reset_done, jnp.zeros_like(x), x), act_hs)

            for agent in range(n_agents):
                obs_i = obs[:, step * n_agents + agent, jnp.newaxis]
                dones_i = dones[:, step * n_agents + agent, jnp.newaxis]
                x, act_hs = rec_apply(params, obs_i, obs_i, obs_i, act_hs, dones_i)
                act_output.append(x)

    act_output = jnp.concatenate(act_output, axis=1)

    learn_output, _ = ret.apply(params, obs, obs, obs, init_hs, dones, 0, method=ret.chunkwise)
    # learn_output = ret.apply(params, obs, obs, obs, dones)

    total_error = jnp.mean(jnp.abs(learn_output - act_output))
    elem_error = jnp.abs(learn_output - act_output)

    assert total_error < TOL
    assert jnp.all(elem_error < TOL)