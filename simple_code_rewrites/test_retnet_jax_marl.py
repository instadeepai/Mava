# test_retnet_jax_rl_multi_agent_parallel_v3_vs_chunkwise_v3.py
import jax
import jax.numpy as jnp
from jax import random
from jax.lax import scan, dynamic_slice
import numpy as np

# Optional: Use float32 for consistency
# jax.config.update("jax_enable_x64", False)

def swish(x):
    """Applies the Swish activation function in JAX."""
    return x * jax.nn.sigmoid(x)

# --- JAX RL Multi-Agent Masking Helpers for Chunkwise v3 ---

def create_decay_matrix_rl_jax_v3(gamma: float, size_ts: int, dones_in_chunk_ts: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """
    Creates the timestep-based decay matrix first, then repeats. Corrected v3.
    size_ts: Number of timesteps in the chunk.
    dones_in_chunk_ts: Dones for the timesteps in this chunk.
    """
    n_ts = jnp.arange(size_ts)
    m_ts = jnp.arange(size_ts)
    k_ts = jnp.arange(size_ts) # Timestep index for checking resets, relative to chunk start (0 to size_ts-1)

    n_ts_grid, m_ts_grid, k_ts_grid = jnp.meshgrid(n_ts, m_ts, k_ts, indexing='ij')

    # 1. Initial decay based on timestep difference
    decay_exponent = jnp.maximum(0, n_ts_grid[:, :, 0] - m_ts_grid[:, :, 0])
    D_ts = jnp.exp(gamma * decay_exponent.astype(jnp.float32))

    # 2. Timestep causality (ts_n < ts_m)
    causal_mask_ts = n_ts_grid[:, :, 0] < m_ts_grid[:, :, 0]
    D_ts = jnp.where(causal_mask_ts, 0.0, D_ts)

    # 3. RL episode boundaries (based on shared dones_ts within chunk)
    # Use dones_ts_prev logic, aligned with the parallel matrix version
    dones_prev_in_chunk = jnp.concatenate([jnp.array([False]), dones_in_chunk_ts[:-1]]) # Assuming False before chunk start relative done
    dones_prev_bc = jnp.reshape(dones_prev_in_chunk, (1, 1, size_ts))

    # k_ts_grid represents the timestep index (0..size_ts-1) AT which a reset might occur
    # Reset occurs at k_ts if dones_prev_in_chunk[k_ts] is True
    reset_condition_per_k_ts = (
        dones_prev_bc[0, 0, k_ts_grid] &  # Check done status of timestep k_ts-1 (relative)
        (n_ts_grid >= k_ts_grid) &        # ts_n is at or after the reset timestep k_ts
        (m_ts_grid < k_ts_grid)           # ts_m is strictly before the reset timestep k_ts
    )
    zeroing_mask_nm_ts = jnp.any(reset_condition_per_k_ts, axis=2) # Check if reset happens between m and n
    D_ts_final = jnp.where(zeroing_mask_nm_ts, 0.0, D_ts) # Shape (size_ts, size_ts)


    # 4. Repeat to create the full chunk_size x chunk_size matrix
    chunk_size = size_ts * num_agents
    D_final = D_ts_final.repeat(num_agents, axis=0).repeat(num_agents, axis=1)

    # 5. Apply fine-grained causality (n < m) for steps within timesteps
    n = jnp.arange(chunk_size)
    m = jnp.arange(chunk_size)
    n_full_grid, m_full_grid = jnp.meshgrid(n, m, indexing='ij')
    fine_causal_mask = n_full_grid < m_full_grid
    D_final = jnp.where(fine_causal_mask, 0.0, D_final)

    return D_final


def create_cross_chunk_mask_rl_jax_v3(size_ts: int, dones_in_chunk_ts: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """
    Creates a mask based on first done timestep (v3 uses dones_ts_prev logic).
    Mask is 1 until the START of the timestep AFTER the first done.
    """
    # Check where dones_in_chunk_ts is True (means reset happens at start of NEXT ts)
    done_indices_local = jnp.where(dones_in_chunk_ts, jnp.arange(size_ts), -1)
    first_done_ts_local = jnp.min(jnp.where(done_indices_local != -1, done_indices_local, size_ts))

    # If first_done_ts_local is size_ts, no done occurred in chunk.
    # If first_done_ts_local is k, reset happens at start of local timestep k+1.
    # Mask should be 1 up to step index (k+1)*num_agents - 1.
    # The first step index to be masked is (first_done_ts_local + 1) * num_agents.
    first_reset_step_index = (first_done_ts_local + 1) * num_agents

    mask = jnp.arange(size_ts * num_agents) < first_reset_step_index
    return mask.astype(jnp.float32)

# --- JAX RL Multi-Agent Chunkwise Implementation (Corrected v3) ---
def chunkwise_retention_rl_multi_agent_jax_v3( # Renamed v3
    Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray,
    dones_ts: jnp.ndarray, # Shape (NUM_TIMESTEPS,)
    gamma: float, chunk_size: int, num_agents: int,
    W_g: jnp.ndarray, W_o: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes Retention using the chunkwise formulation with JAX for multi-agent RL (shared dones v3).
    Assumes chunk_size is a multiple of num_agents. Scan carry fixed, masking refined.
    """
    seq_len, embed_dim = Q.shape
    num_timesteps = dones_ts.shape[0]
    if seq_len != num_timesteps * num_agents:
         raise ValueError("seq_len must equal num_timesteps * num_agents")
    if chunk_size % num_agents != 0:
        raise ValueError("chunk_size must be divisible by num_agents for multi-agent chunkwise")

    num_chunks = seq_len // chunk_size
    timesteps_per_chunk = chunk_size // num_agents
    scaling = embed_dim**-0.5

    # --- Precomputation ---
    chunk_indices_ts = jnp.arange(timesteps_per_chunk, dtype=jnp.float32)
    # xi[n] = gamma ^ ts_n_local
    xi_vec = jnp.exp(gamma * chunk_indices_ts).repeat(num_agents, axis=0)
    # zeta[m] = gamma ^ (T_chunk - 1 - ts_m_local)
    zeta_vec = jnp.exp(gamma * (timesteps_per_chunk - 1 - chunk_indices_ts)).repeat(num_agents, axis=0)
    # gamma_B decays state over T_chunk timesteps
    gamma_B = jnp.exp(gamma * timesteps_per_chunk)
    K_scaled = K * scaling

    # --- Reshape ---
    Q_c = Q.reshape(num_chunks, chunk_size, embed_dim)
    K_c_scaled = K_scaled.reshape(num_chunks, chunk_size, embed_dim)
    V_c = V.reshape(num_chunks, chunk_size, embed_dim)

    # --- Scan ---
    R_initial = jnp.zeros((embed_dim, embed_dim), dtype=jnp.float32)
    initial_carry = (R_initial, jnp.array(False)) # (R_state, prev_chunk_last_ts_done)
    scan_inputs = (Q_c, K_c_scaled, V_c, jnp.arange(num_chunks)) # Pass chunk index

    def scan_body_chunkwise_rl_ma_shared_v3(carry, x_chunk_data):
        R_in_state, prev_chunk_last_ts_done = carry
        q_i, k_i_scaled, v_i, chunk_idx_i = x_chunk_data

        # --- Slice Dones for Chunk ---
        ts_start_chunk = chunk_idx_i * timesteps_per_chunk
        safe_ts_start_chunk = jnp.clip(ts_start_chunk, 0, num_timesteps - timesteps_per_chunk) # Avoid OOB
        # Dones for timesteps within this chunk
        dones_in_chunk_ts = dynamic_slice(dones_ts, (safe_ts_start_chunk,), (timesteps_per_chunk,))

        # --- Calculate Chunk Output ---
        R_in_actual = jnp.where(prev_chunk_last_ts_done, jnp.zeros_like(R_in_state), R_in_state)

        # Use corrected masking functions
        D_i = create_decay_matrix_rl_jax_v3(gamma, timesteps_per_chunk, dones_in_chunk_ts, num_agents)
        inner_qk_i = (q_i @ k_i_scaled.T) * D_i
        inner_output_i = inner_qk_i @ v_i

        cross_qr_i = q_i @ R_in_actual
        cross_mask_i = create_cross_chunk_mask_rl_jax_v3(timesteps_per_chunk, dones_in_chunk_ts, num_agents)
        cross_output_i = cross_qr_i * xi_vec[:, None] * cross_mask_i[:, None]
        chunk_output = inner_output_i + cross_output_i

        # --- Calculate State R_out for Next Chunk ---
        kv_zeta_i_full = k_i_scaled.T @ (v_i * zeta_vec[:, None])

        # Find last done index RELATIVE to the start of the chunk's timesteps
        done_ts_indices_local = jnp.where(dones_in_chunk_ts, jnp.arange(timesteps_per_chunk), -1)
        last_done_ts_local = jnp.max(done_ts_indices_local) # Max local index (-1 if none)

        def calculate_R_out_reset():
            # Reset occurred within the chunk. State only includes contributions AFTER the reset.
            # start_n_relative is the first step index (0 to chunk_size-1) AFTER the reset.
            start_n_relative = (last_done_ts_local + 1) * num_agents
            last_ep_mask = (jnp.arange(chunk_size) >= start_n_relative).astype(jnp.float32)
            k_last_ep = k_i_scaled * last_ep_mask[:, None]
            v_last_ep = v_i * last_ep_mask[:, None]
            # Calculate state only from the last segment, correctly weighted by zeta
            return k_last_ep.T @ (v_last_ep * zeta_vec[:, None])

        def calculate_R_out_no_reset():
            # No reset in this chunk. Decay incoming state, add full chunk contribution.
            return gamma_B * R_in_actual + kv_zeta_i_full

        R_out_state = jax.lax.cond(
            last_done_ts_local == -1, # Condition: No reset occurred within this chunk's timesteps
            calculate_R_out_no_reset,
            calculate_R_out_reset
        )

        # --- Update Carry: Get done status of the last timestep of *this* chunk ---
        current_chunk_last_ts_idx = ts_start_chunk + timesteps_per_chunk - 1
        safe_idx = jnp.clip(current_chunk_last_ts_idx, 0, num_timesteps - 1)
        current_chunk_last_ts_done = jnp.where(
            current_chunk_last_ts_idx >= 0,
            dynamic_slice(dones_ts, (safe_idx,), (1,))[0],
            False
        )
        new_carry = (R_out_state, current_chunk_last_ts_done) # <-- Correctly update carry

        return new_carry, chunk_output

    # Run the scan
    final_carry, chunk_outputs_all = scan(scan_body_chunkwise_rl_ma_shared_v3, initial_carry, scan_inputs)
    retention_result = chunk_outputs_all.reshape(seq_len, embed_dim)
    gating_signal = K @ W_g
    gated_output = (swish(gating_signal) * retention_result) @ W_o
    return gated_output

# --- JAX RL Multi-Agent Parallel Implementation (v2) ---
def create_full_decay_matrix_rl_ma_jax_v2(
    gamma: float, seq_len: int, num_agents: int,
    dones_ts: jnp.ndarray # Shape (NUM_TIMESTEPS,)
) -> jnp.ndarray:
    """
    Creates the full (SEQ_LEN, SEQ_LEN) decay matrix D for multi-agent parallel computation.
    Decay happens per timestep, resets based on dones_ts[ts-1]. (JIT-friendly - v2 Reset Logic)
    """
    num_timesteps = dones_ts.shape[0]
    if seq_len != num_timesteps * num_agents:
        raise ValueError("seq_len must equal num_timesteps * num_agents")

    n = jnp.arange(seq_len)
    m = jnp.arange(seq_len)
    # Index for checking reset timesteps, goes from 1 to num_timesteps
    ts_reset_check = jnp.arange(1, num_timesteps + 1) # Timestep index where reset occurs

    n_grid, m_grid = jnp.meshgrid(n, m, indexing='ij')
    n_grid_bc = n_grid[:, :, None]
    m_grid_bc = m_grid[:, :, None]
    tsr_grid = jnp.reshape(ts_reset_check, (1, 1, num_timesteps)) # Shape (1, 1, num_timesteps)

    ts_n = n_grid // num_agents
    ts_m = m_grid // num_agents
    decay_exponent = jnp.maximum(0, ts_n - ts_m)
    D = jnp.exp(gamma * decay_exponent.astype(jnp.float32))

    causal_mask = n_grid < m_grid
    D = jnp.where(causal_mask, 0.0, D)

    dones_ts_prev = jnp.concatenate([jnp.array([False]), dones_ts[:-1]]) # Dones that trigger reset
    dones_prev_bc = jnp.reshape(dones_ts_prev, (1, 1, num_timesteps)) # Shape (1, 1, num_timesteps)
    ts_n_bc = n_grid_bc // num_agents # Timestep index (0 to T-1) for step n
    ts_m_bc = m_grid_bc // num_agents # Timestep index (0 to T-1) for step m

    # Corrected logic: reset if n is in or after timestep ts_r (index ts_r-1)
    # and m is strictly before timestep ts_r (index ts_r-1),
    # and the timestep ts_r-1 was done.
    # tsr_grid runs 1..T, so tsr_grid-1 runs 0..T-1 (matching dones_ts_prev indices)
    reset_condition_per_tsr = (
        dones_prev_bc[0, 0, tsr_grid-1] & # Check dones_ts[ts_r-1]
        (ts_n_bc >= (tsr_grid-1)) &       # ts_n is >= timestep index where reset happens
        (ts_m_bc < (tsr_grid-1))          # ts_m is < timestep index where reset happens
    )

    reset_mask_nm = jnp.any(reset_condition_per_tsr, axis=2)
    D_final = jnp.where(reset_mask_nm, 0.0, D)

    return D_final

def parallel_retention_rl_multi_agent_jax_v2(
    Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray,
    dones_ts: jnp.ndarray, # Shape (NUM_TIMESTEPS,)
    gamma: float, num_agents: int,
    W_g: jnp.ndarray, W_o: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes Retention using the parallel formulation (v2 Reset Logic)
    for multi-agent RL with shared dones.
    """
    seq_len, embed_dim = Q.shape
    num_timesteps = dones_ts.shape[0]
    if seq_len != num_timesteps * num_agents:
        raise ValueError("seq_len must equal num_timesteps * num_agents")

    scaling = embed_dim**-0.5
    K_scaled = K * scaling

    D_matrix = create_full_decay_matrix_rl_ma_jax_v2(gamma, seq_len, num_agents, dones_ts)
    QK_scaled = Q @ K_scaled.T
    retention_result = (QK_scaled * D_matrix) @ V

    gating_signal = K @ W_g
    gated_output = (swish(gating_signal) * retention_result) @ W_o
    return gated_output


# --- Simulation Parameters ---
NUM_TIMESTEPS = 16
NUM_AGENTS = 4
SEQ_LEN = NUM_TIMESTEPS * NUM_AGENTS # 64
EMBED_DIM = 128
CHUNK_SIZE = NUM_AGENTS * 2 # Process 2 timesteps per chunk = 8 steps total
GAMMA = jnp.log(0.9).astype(jnp.float32)

# --- Generate Dummy Data & Weights ---
key = random.PRNGKey(42)
keys = random.split(key, 5)
Q_data = random.normal(keys[0], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
K_data = random.normal(keys[1], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
V_data = random.normal(keys[2], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
W_g_data = random.normal(keys[3], (EMBED_DIM, EMBED_DIM), dtype=jnp.float32) * (1 / EMBED_DIM)
W_o_data = random.normal(keys[4], (EMBED_DIM, EMBED_DIM), dtype=jnp.float32) * (1 / EMBED_DIM)

# --- Generate Dummy Dones ---
dones_np_ts = np.zeros(NUM_TIMESTEPS, dtype=bool)
dones_np_ts[5] = True
dones_np_ts[10] = True
dones_ts_data = jnp.array(dones_np_ts)

print(f"Num Timesteps: {NUM_TIMESTEPS}, Num Agents: {NUM_AGENTS}, Seq Len: {SEQ_LEN}, Chunk Size: {CHUNK_SIZE}")
print("Dones (Per Timestep):", dones_ts_data.astype(int))
print("-" * 20)

# --- Run JIT-compiled Formulations ---
chunkwise_retention_rl_multi_agent_jax_jit_v3 = jax.jit(chunkwise_retention_rl_multi_agent_jax_v3, static_argnums=(5, 6)) # chunk_size, num_agents
parallel_retention_rl_multi_agent_jax_jit_v2 = jax.jit(parallel_retention_rl_multi_agent_jax_v2, static_argnums=(5,)) # num_agents

print("\nRunning Chunkwise v3 Retention (RL MA Shared Dones JAX JIT)...")
chunkwise_output_rl_ma = chunkwise_retention_rl_multi_agent_jax_jit_v3(
    Q_data, K_data, V_data, dones_ts_data, GAMMA, CHUNK_SIZE, NUM_AGENTS, W_g_data, W_o_data
).block_until_ready()

print("\nRunning Parallel v2 Retention (RL MA Shared Dones JAX JIT)...")
parallel_output_rl_ma = parallel_retention_rl_multi_agent_jax_jit_v2(
    Q_data, K_data, V_data, dones_ts_data, GAMMA, NUM_AGENTS, W_g_data, W_o_data
).block_until_ready()


# --- Compare Results ---
print("\nComparing RL MA Shared Dones JAX outputs (Chunkwise v3 vs Parallel v2)...")
print(f"Chunkwise v3 RL MA JAX Output Shape: {chunkwise_output_rl_ma.shape}")
print(f"Parallel v2 RL MA JAX Output Shape: {parallel_output_rl_ma.shape}")

# Use a stricter tolerance now
are_close_rl_ma = jnp.allclose(chunkwise_output_rl_ma, parallel_output_rl_ma, atol=1e-6, rtol=1e-6)

print(f"\nRL MA Shared Dones JAX Outputs are equivalent: {are_close_rl_ma}")

if not are_close_rl_ma:
    print("Difference between RL MA Shared Dones JAX outputs (Chunkwise vs Parallel):")
    diff = chunkwise_output_rl_ma - parallel_output_rl_ma
    max_diff = jnp.max(jnp.abs(diff))
    print(f"Maximum absolute difference: {max_diff}")
    mean_diff = jnp.mean(jnp.abs(diff))
    print(f"Mean absolute difference: {mean_diff}")

    diff_indices = jnp.where(~jnp.isclose(chunkwise_output_rl_ma, parallel_output_rl_ma, atol=1e-6, rtol=1e-6))
    print("First few differing indices (row, col):")
    num_diff_to_show = min(10, len(diff_indices[0]))
    for i in range(num_diff_to_show):
        r, c = diff_indices[0][i], diff_indices[1][i]
        chk_val = chunkwise_output_rl_ma[r, c]
        par_val = parallel_output_rl_ma[r, c]
        print(f"  Index ({r}, {c}): Chunk={chk_val:.8f}, Parallel={par_val:.8f}, Diff={chk_val - par_val:.4e}")