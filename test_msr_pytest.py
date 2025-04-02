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
import pytest
from omegaconf import DictConfig

from mava.networks.retention import MultiScaleRetention


@pytest.fixture
def test_parameters():
    """Test parameters for MultiScaleRetention tests."""
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
            "decay_scaling_factor": 0.3,
            "timestep_positional_encoding": True,
            "timestep_chunk_size": None,
        }
    )

    decay_kappas = 1 - jnp.exp(jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), retnet_num_heads))
    decay_kappas *= memory_config.decay_scaling_factor
    decay_kappas = jnp.log(decay_kappas)
    decay_kappas = decay_kappas[None, :, None, None]

    # Generate random key and observations
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs = jax.random.normal(subkey, (bsz, seq_len, retnet_embed_dim))

    # Initialize step counts
    step_counts = jnp.arange(num_time_steps)
    step_counts = step_counts[None, ...].repeat(bsz, axis=0)[..., None].repeat(num_agents, axis=-1)
    step_counts = step_counts.reshape(bsz, seq_len)

    # Initial hidden state and scale
    init_hstate = jnp.zeros(
        (
            bsz,
            retnet_num_heads,
            retnet_embed_dim // retnet_num_heads,
            retnet_embed_dim // retnet_num_heads,
        )
    )
    init_scale = jnp.ones((bsz, retnet_num_heads, 1, 1))

    # No resets (all zeros)
    no_dones = jnp.zeros((bsz, seq_len), dtype=bool)

    # Generate random dones
    key, done_key = jax.random.split(key)
    random_dones = jnp.repeat(
        jax.random.randint(done_key, (bsz, num_time_steps), 0, 2).astype(bool), num_agents, axis=1
    )

    return {
        "bsz": bsz,
        "num_agents": num_agents,
        "obs_dim": obs_dim,
        "num_time_steps": num_time_steps,
        "seq_len": seq_len,
        "retnet_embed_dim": retnet_embed_dim,
        "retnet_num_heads": retnet_num_heads,
        "num_chunks": num_chunks,
        "memory_config": memory_config,
        "decay_kappas": decay_kappas,
        "key": key,
        "obs": obs,
        "step_counts": step_counts,
        "init_hstate": init_hstate,
        "init_scale": init_scale,
        "no_dones": no_dones,
        "random_dones": random_dones,
    }


def run_encoder_recurrent(
    params,
    obs,
    step_counts,
    init_hstate,
    init_scale,
    dones,
    num_agents,
    num_time_steps,
    decay_kappas,
    enc_jit_inf,
):
    """Run encoder in recurrent form step by step."""
    hstate = copy.deepcopy(init_hstate)
    scale = copy.deepcopy(init_scale)
    act_output = []

    for step in range(num_time_steps):
        new_scale = scale * jnp.exp(decay_kappas) + 1.0
        hstate_scale_factor = jnp.sqrt(scale) * jnp.exp(decay_kappas) / jnp.sqrt(new_scale)
        hstate = hstate * hstate_scale_factor
        scale = new_scale

        # Handle dones if they exist
        if jnp.any(dones):
            reset_done = dones[:, step * num_agents, None, None, None]
            hstate = jax.tree.map(
                lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.zeros_like(x), x), hstate
            )
            scale = jax.tree.map(
                lambda x, reset_done=reset_done: jnp.where(reset_done, jnp.ones_like(x), x), scale
            )

        obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
        step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

        out, hstate = enc_jit_inf(
            params,
            obs_i,
            obs_i,
            obs_i,
            hstate,
            step_counts_i,
            kv_scale=scale,
        )
        act_output.append(out)

    return jnp.concatenate(act_output, axis=1)


def run_decoder_recurrent(
    params,
    obs,
    step_counts,
    init_hstate,
    init_scale,
    dones,
    num_agents,
    num_time_steps,
    decay_kappas,
    dec_jit_inf,
):
    """Run decoder in recurrent form step by step for each agent."""
    hstate = copy.deepcopy(init_hstate)
    scale = copy.deepcopy(init_scale)
    act_output = []

    for step in range(num_time_steps):
        obs_i = obs[:, step * num_agents : (step + 1) * num_agents, ...]
        step_counts_i = step_counts[:, step * num_agents : (step + 1) * num_agents]

        new_scale = scale * jnp.exp(decay_kappas) + 1.0
        hstate_scale_factor = jnp.sqrt(scale) * jnp.exp(decay_kappas) / jnp.sqrt(new_scale)
        hstate = hstate * hstate_scale_factor
        scale = new_scale

        # Handle dones if they exist
        if jnp.any(dones):
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
                params,
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

    return jnp.concatenate(act_output, axis=1)


def test_unmasked_msr_no_dones(test_parameters):
    """Test unmasked MSR without done signals."""
    # Extract parameters
    params = test_parameters

    # Create unmasked MSR
    msr_enc = MultiScaleRetention(
        embed_dim=params["retnet_embed_dim"],
        n_head=params["retnet_num_heads"],
        n_agents=params["num_agents"],
        memory_config=params["memory_config"],
        masked=False,
        decay_scaling_factor=params["memory_config"].decay_scaling_factor,
    )

    # Initialize parameters
    init_key = jax.random.PRNGKey(42)
    msr_enc_params = msr_enc.init(
        init_key,
        params["obs"][0:1, 0:1, ...],
        params["obs"][0:1, 0:1, ...],
        params["obs"][0:1, 0:1, ...],
        params["init_hstate"][0:1, ...],
        params["step_counts"][0:1, 0:1],
        params["init_scale"][0:1, ...],
        method="recurrent",
    )

    # Create jitted functions
    enc_jit_apply = partial(jax.jit(msr_enc.apply, static_argnames=("num_chunks", "inference")))
    enc_jit_inf = jax.jit(partial(msr_enc.apply, method="recurrent"))

    # Run step-by-step recurrent form
    act_output = run_encoder_recurrent(
        msr_enc_params,
        params["obs"],
        params["step_counts"],
        params["init_hstate"],
        params["init_scale"],
        params["no_dones"],
        params["num_agents"],
        params["num_time_steps"],
        params["decay_kappas"],
        enc_jit_inf,
    )

    # Run all-at-once chunkwise form
    train_out, _, _ = enc_jit_apply(
        msr_enc_params,
        params["obs"],
        params["obs"],
        params["obs"],
        params["init_hstate"],
        params["no_dones"],
        params["step_counts"],
        num_chunks=params["num_chunks"],
        kv_scale=params["init_scale"],
        inference=False,
    )

    # Calculate error and assert it's within threshold
    total_error = jnp.mean(jnp.abs(train_out - act_output))
    assert 1e-7 < total_error < 1e-5, f"Error: {total_error} exceeds threshold of 1e-6"


def test_masked_msr_no_dones(test_parameters):
    """Test masked MSR without done signals."""
    # Extract parameters
    params = test_parameters

    # Create masked MSR
    msr_dec = MultiScaleRetention(
        embed_dim=params["retnet_embed_dim"],
        n_head=params["retnet_num_heads"],
        n_agents=params["num_agents"],
        memory_config=params["memory_config"],
        masked=True,
        decay_scaling_factor=params["memory_config"].decay_scaling_factor,
    )

    # Initialize parameters
    init_key = jax.random.PRNGKey(43)
    msr_dec_params = msr_dec.init(
        init_key,
        params["obs"][0:1, 0:1, ...],
        params["obs"][0:1, 0:1, ...],
        params["obs"][0:1, 0:1, ...],
        params["init_hstate"][0:1, ...],
        params["step_counts"][0:1, 0:1],
        params["init_scale"][0:1, ...],
        method="recurrent",
    )

    # Create jitted functions
    dec_jit_apply = partial(jax.jit(msr_dec.apply, static_argnames=("num_chunks", "inference")))
    dec_jit_inf = jax.jit(partial(msr_dec.apply, method="recurrent"))

    # Run step-by-step recurrent form with agent loop
    act_output = run_decoder_recurrent(
        msr_dec_params,
        params["obs"],
        params["step_counts"],
        params["init_hstate"],
        params["init_scale"],
        params["no_dones"],
        params["num_agents"],
        params["num_time_steps"],
        params["decay_kappas"],
        dec_jit_inf,
    )

    # Run all-at-once chunkwise form
    train_out, _, _ = dec_jit_apply(
        msr_dec_params,
        params["obs"],
        params["obs"],
        params["obs"],
        params["init_hstate"],
        params["no_dones"],
        params["step_counts"],
        num_chunks=params["num_chunks"],
        kv_scale=params["init_scale"],
        inference=False,
    )

    # Calculate error and assert it's within threshold
    total_error = jnp.mean(jnp.abs(train_out - act_output))
    assert 1e-7 < total_error < 1e-5, f"Error: {total_error} exceeds threshold of 1e-6"


def test_unmasked_msr_with_dones(test_parameters):
    """Test unmasked MSR with done signals."""
    # Extract parameters
    params = test_parameters

    # Create unmasked MSR (reuse the same instance as in test 1)
    msr_enc = MultiScaleRetention(
        embed_dim=params["retnet_embed_dim"],
        n_head=params["retnet_num_heads"],
        n_agents=params["num_agents"],
        memory_config=params["memory_config"],
        masked=False,
        decay_scaling_factor=params["memory_config"].decay_scaling_factor,
    )

    # Initialize parameters
    init_key = jax.random.PRNGKey(44)
    msr_enc_params = msr_enc.init(
        init_key,
        params["obs"][0:1, 0:1, ...],
        params["obs"][0:1, 0:1, ...],
        params["obs"][0:1, 0:1, ...],
        params["init_hstate"][0:1, ...],
        params["step_counts"][0:1, 0:1],
        params["init_scale"][0:1, ...],
        method="recurrent",
    )

    # Create jitted functions
    enc_jit_apply = partial(jax.jit(msr_enc.apply, static_argnames=("num_chunks", "inference")))
    enc_jit_inf = jax.jit(partial(msr_enc.apply, method="recurrent"))

    # Run step-by-step recurrent form with dones
    act_output = run_encoder_recurrent(
        msr_enc_params,
        params["obs"],
        params["step_counts"],
        params["init_hstate"],
        params["init_scale"],
        params["random_dones"],
        params["num_agents"],
        params["num_time_steps"],
        params["decay_kappas"],
        enc_jit_inf,
    )

    # Run all-at-once chunkwise form with dones
    train_out, _, _ = enc_jit_apply(
        msr_enc_params,
        params["obs"],
        params["obs"],
        params["obs"],
        params["init_hstate"],
        params["random_dones"],
        params["step_counts"],
        num_chunks=params["num_chunks"],
        kv_scale=params["init_scale"],
        inference=False,
    )

    # Calculate error and assert it's within threshold
    total_error = jnp.mean(jnp.abs(train_out - act_output))
    assert 1e-7 < total_error < 1e-5, f"Error: {total_error} exceeds threshold of 1e-6"


def test_masked_msr_with_dones(test_parameters):
    """Test masked MSR with done signals."""
    # Extract parameters
    params = test_parameters

    # Create masked MSR (reuse the same instance as in test 2)
    msr_dec = MultiScaleRetention(
        embed_dim=params["retnet_embed_dim"],
        n_head=params["retnet_num_heads"],
        n_agents=params["num_agents"],
        memory_config=params["memory_config"],
        masked=True,
        decay_scaling_factor=params["memory_config"].decay_scaling_factor,
    )

    # Initialize parameters
    init_key = jax.random.PRNGKey(45)
    msr_dec_params = msr_dec.init(
        init_key,
        params["obs"][0:1, 0:1, ...],
        params["obs"][0:1, 0:1, ...],
        params["obs"][0:1, 0:1, ...],
        params["init_hstate"][0:1, ...],
        params["step_counts"][0:1, 0:1],
        params["init_scale"][0:1, ...],
        method="recurrent",
    )

    # Create jitted functions
    dec_jit_apply = partial(jax.jit(msr_dec.apply, static_argnames=("num_chunks", "inference")))
    dec_jit_inf = jax.jit(partial(msr_dec.apply, method="recurrent"))

    # Run step-by-step recurrent form with agent loop and dones
    act_output = run_decoder_recurrent(
        msr_dec_params,
        params["obs"],
        params["step_counts"],
        params["init_hstate"],
        params["init_scale"],
        params["random_dones"],
        params["num_agents"],
        params["num_time_steps"],
        params["decay_kappas"],
        dec_jit_inf,
    )

    # Run all-at-once chunkwise form with dones
    train_out, _, _ = dec_jit_apply(
        msr_dec_params,
        params["obs"],
        params["obs"],
        params["obs"],
        params["init_hstate"],
        params["random_dones"],
        params["step_counts"],
        num_chunks=params["num_chunks"],
        kv_scale=params["init_scale"],
        inference=False,
    )

    # Calculate error and assert it's within threshold
    total_error = jnp.mean(jnp.abs(train_out - act_output))
    assert 1e-7 < total_error < 1e-5, f"Error: {total_error} exceeds threshold of 1e-6"
