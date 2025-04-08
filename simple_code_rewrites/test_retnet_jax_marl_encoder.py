import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.lax import scan

np.set_printoptions(edgeitems=30, linewidth = 1000000)
jnp.set_printoptions(edgeitems=30, linewidth = 1000000)

def swish(x):
    """Applies the Swish activation function in JAX."""
    return x * jax.nn.sigmoid(x)

# --- JAX RL Masking Helpers (Revised create_decay_matrix) ---

def create_decay_matrix_rl_jax(gamma: float, size: int, dones_in_chunk: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """
    Creates the intra-chunk decay matrix D in JAX, respecting done boundaries. (JIT-friendly)
    """
    n = jnp.arange(size)
    m = jnp.arange(size)
    k = jnp.arange(size) # Dimension for checking done indices

    # Create broadcastable grids
    n_grid, m_grid, k_grid = jnp.meshgrid(n, m, k, indexing='ij')

    # 1. Initial decay based on relative position (n, m)
    # Calculate diff only once using slices to avoid redundant dimension
    diff = n_grid[:, :, 0] - m_grid[:, :, 0]
    D = jnp.exp(gamma * diff).astype(jnp.float32)

    # 2. Standard causality (n < m)
    causal_mask = n_grid[:, :, 0] < m_grid[:, :, 0]
    D = jnp.where(causal_mask, 0.0, D)

    # 3. RL episode boundaries
    # Condition for zeroing D[n, m]: exists k such that (dones[k] AND n > k AND m <= k)
    # Broadcast dones_in_chunk to shape (1, 1, size) to align with k_grid
    dones_bc = jnp.reshape(dones_in_chunk, (1, 1, size))

    # Check the condition across all k for each (n, m) pair
    zero_condition_per_k = (dones_bc & (n_grid > k_grid) & (m_grid <= k_grid))

    # Find if the condition is True for *any* k for each (n, m)
    zeroing_mask_nm = jnp.any(zero_condition_per_k, axis=2) # Reduce along the k axis

    # Apply the zeroing mask to D
    D_final = jnp.where(zeroing_mask_nm, 0.0, D)
    D_final = D_final.repeat(num_agents, axis=0).repeat(num_agents, axis=1)

    # D_final = jnp.tril(D_final)

    return D_final


def create_cross_chunk_mask_rl_jax(size: int, dones_in_chunk: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """
    Creates a mask in JAX to zero out contributions after a done flag within the chunk.
    (Remains the same as previous version)
    """
    first_done_idx = jnp.argmax(dones_in_chunk)
    done_occurred = dones_in_chunk[first_done_idx]
    valid_first_done_idx = jnp.where(done_occurred, first_done_idx, size)
    mask = jnp.arange(size) <= valid_first_done_idx
    mask = mask.repeat(num_agents, axis=0)
    return mask.astype(jnp.float32)


# --- JAX RL Retention Implementations (No changes needed here) ---

def recurrent_retention_rl_jax(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray, dones: jnp.ndarray, gamma: float, W_g: jnp.ndarray, W_o: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """
    Computes Retention using the recurrent formulation with JAX lax.scan,
    handling RL episode boundaries. Includes K scaling and Swish gating.
    """
    seq_len, embed_dim = Q.shape
    # if dones.shape != (seq_len,):
    #     raise ValueError(f"Dones shape mismatch. Expected ({seq_len},), got {dones.shape}")

    scaling = embed_dim**-0.5
    K_scaled = K * scaling
    dones_prev = jnp.concatenate([jnp.array([False]), dones[:-1]])
    inputs = (Q.reshape(seq_len // num_agents, num_agents, embed_dim), K_scaled.reshape(seq_len // num_agents, num_agents, embed_dim), V.reshape(seq_len // num_agents, num_agents, embed_dim), dones_prev)

    def scan_body_agents(carry_S, x_t):
        k_agent_scaled, v_agent = x_t
        carry_S += jnp.outer(k_agent_scaled, v_agent)
        return carry_S, None
    
    def scan_body_recurrent_rl(carry_S, x_t):
        q_n, k_n_scaled, v_n, done_n_prev = x_t
        S_reset_or_not = jnp.where(done_n_prev, jnp.zeros_like(carry_S), carry_S)
        new_S = jnp.exp(gamma) * S_reset_or_not
        new_S, _ = scan(scan_body_agents, new_S, (k_n_scaled, v_n))
        ret_n = q_n @ new_S
        return new_S, ret_n

    S_initial = jnp.zeros((embed_dim, embed_dim), dtype=jnp.float32)
    _, retention_result = scan(scan_body_recurrent_rl, S_initial, inputs)
    gating_signal = K @ W_g
    gated_output = (swish(gating_signal) * retention_result.reshape(seq_len, embed_dim)) @ W_o
    return gated_output


def chunkwise_retention_rl_jax(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray, dones: jnp.ndarray, gamma: float, chunk_size: int, W_g: jnp.ndarray, W_o: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """
    Computes Retention using the chunkwise formulation with JAX, handling RL boundaries.
    Includes K scaling, Swish gating, and refined state update logic. Uses lax.scan.
    """
    seq_len, embed_dim = Q.shape
    if seq_len % chunk_size != 0:
        raise ValueError("Sequence length must be divisible by chunk_size for chunkwise RL")
    # if dones.shape != (seq_len,):
    #     raise ValueError(f"Dones shape mismatch. Expected ({seq_len},), got {dones.shape}")

    num_chunks = seq_len // chunk_size
    num_timesteps = seq_len // num_agents
    done_chunk_size = num_timesteps // num_chunks
    scaling = embed_dim**-0.5

    xi_vec = jnp.exp(gamma * (jnp.arange(done_chunk_size, dtype=jnp.float32) + 1)).repeat(num_agents, axis=0)
    zeta_vec = jnp.exp(gamma * (done_chunk_size - 1 - jnp.arange(done_chunk_size, dtype=jnp.float32))).repeat(num_agents, axis=0)
    gamma_B = jnp.exp(gamma * done_chunk_size)
    K_scaled = K * scaling

    Q_c = Q.reshape(num_chunks, chunk_size, embed_dim)
    K_c_scaled = K_scaled.reshape(num_chunks, chunk_size, embed_dim)
    V_c = V.reshape(num_chunks, chunk_size, embed_dim)
    dones_c = dones.reshape(num_chunks, done_chunk_size)

    R_initial = jnp.zeros((embed_dim, embed_dim), dtype=jnp.float32)
    scan_inputs = (Q_c, K_c_scaled, V_c, dones_c)

    def scan_body_chunkwise_rl(carry_R_in, x_chunk):
        q_i, k_i_scaled, v_i, dones_i = x_chunk
        D_i = create_decay_matrix_rl_jax(gamma, done_chunk_size, dones_i, num_agents) # Use JIT-friendly version
        inner_qk_i = (q_i @ k_i_scaled.T) * D_i
        inner_output_i = inner_qk_i @ v_i
        cross_qr_i = q_i @ carry_R_in
        cross_mask_i = create_cross_chunk_mask_rl_jax(done_chunk_size, dones_i, num_agents)
        cross_output_i = cross_qr_i * xi_vec[:, None] * cross_mask_i[:, None]
        chunk_output = inner_output_i + cross_output_i

        # zeta is the last row of D_i
        # zeta_vec = D_i[-1]
        # delta = ~jnp.any(dones_i)
        # R_out = k_i_scaled.T @ (zeta_vec[:, None] * v_i) + delta * gamma_B * carry_R_in
        done_indices_in_chunk = jnp.where(dones_i, jnp.arange(done_chunk_size), -1)
        last_done_index = jnp.max(done_indices_in_chunk)
        start_j = last_done_index + 1
        last_ep_mask = (jnp.arange(done_chunk_size) >= start_j).astype(jnp.float32).repeat(num_agents, axis=0)
        k_last_ep = k_i_scaled * last_ep_mask[:, None]
        v_last_ep = v_i * last_ep_mask[:, None]
        kv_zeta_i_last_ep = k_last_ep.T @ (v_last_ep * zeta_vec[:, None])
        R_out = jnp.where(
            start_j == 0,
            gamma_B * carry_R_in + kv_zeta_i_last_ep,
            kv_zeta_i_last_ep
        )


        return R_out, chunk_output

    final_R, chunk_outputs_all = scan(scan_body_chunkwise_rl, R_initial, scan_inputs)
    retention_result = chunk_outputs_all.reshape(seq_len, embed_dim)
    gating_signal = K @ W_g
    gated_output = (swish(gating_signal) * retention_result) @ W_o
    return gated_output

# --- Simulation Parameters ---
NUM_TIMESTEPS = 16
NUM_AGENTS = 3
SEQ_LEN = NUM_TIMESTEPS * NUM_AGENTS
EMBED_DIM = 128
CHUNK_SIZE = 12
GAMMA = jnp.log(0.5).astype(jnp.float32) # Ensure gamma is float32

# --- Generate Dummy Data & Weights with JAX PRNG ---
key = random.PRNGKey(42)
keys = random.split(key, 5)

Q_data = random.normal(keys[0], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
K_data = random.normal(keys[1], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
V_data = random.normal(keys[2], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
W_g_data = random.normal(keys[3], (EMBED_DIM, EMBED_DIM), dtype=jnp.float32) * (1 / EMBED_DIM)
W_o_data = random.normal(keys[4], (EMBED_DIM, EMBED_DIM), dtype=jnp.float32) * (1 / EMBED_DIM)

# --- Generate Dummy Dones (as JAX array) ---
dones_np = np.zeros(NUM_TIMESTEPS, dtype=bool)
dones_np[5] = True
dones_np[7] = True # End of chunk 1
dones_np[14] = True
dones_data = jnp.array(dones_np)

print("Dones:", dones_data.astype(int))
print("-" * 20)

# --- Run JIT-compiled Formulations ---
recurrent_retention_rl_jax_jit = jax.jit(recurrent_retention_rl_jax, static_argnums=(7,)) # num_agents is static
chunkwise_retention_rl_jax_jit = jax.jit(chunkwise_retention_rl_jax, static_argnums=(5, 8)) # chunk_size and num_agents are static

print("Running Recurrent Retention (RL JAX JIT)...")
recurrent_output_rl = recurrent_retention_rl_jax_jit(Q_data, K_data, V_data, dones_data, GAMMA, W_g_data, W_o_data, NUM_AGENTS).block_until_ready()

print("\nRunning Chunkwise Retention (RL JAX JIT)...")
chunkwise_output_rl = chunkwise_retention_rl_jax_jit(Q_data, K_data, V_data, dones_data, GAMMA, CHUNK_SIZE, W_g_data, W_o_data, NUM_AGENTS).block_until_ready()

# --- Compare Results ---
print("\nComparing RL JAX outputs...")
print(f"Recurrent RL JAX Output Shape: {recurrent_output_rl.shape}")
print(f"Chunkwise RL JAX Output Shape: {chunkwise_output_rl.shape}")

are_close_rl = jnp.allclose(recurrent_output_rl, chunkwise_output_rl, atol=1e-6)

print(f"\nRL JAX Outputs are equivalent (Recurrent RL vs. Chunkwise RL): {are_close_rl}")

if not are_close_rl:
    print("Difference between RL JAX outputs:")
    diff = recurrent_output_rl - chunkwise_output_rl
    max_diff = jnp.max(jnp.abs(diff))
    print(f"Maximum absolute difference: {max_diff}")
    mean_diff = jnp.mean(jnp.abs(diff))
    print(f"Mean absolute difference: {mean_diff}")