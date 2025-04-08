# test_retnet_jax.py
import jax
import jax.numpy as jnp
from jax import random
from jax.lax import scan

# Set float32 for better precision, matching numpy example
# jax.config.update("jax_enable_x64", True)

def swish(x):
    return x * jax.nn.sigmoid(x) # Use jax.nn.sigmoid for potentially better stability

def create_decay_matrix(gamma: float, size: int) -> jnp.ndarray:
    """Creates the intra-chunk decay matrix D using JAX."""
    n = jnp.arange(size)
    m = jnp.arange(size)
    n_grid, m_grid = jnp.meshgrid(n, m, indexing='ij') # Equivalent to np.ogrid
    D = jnp.where(n_grid >= m_grid, jnp.exp(gamma*(n_grid - m_grid)), 0.0)
    return D

def recurrent_retention(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray, gamma: float, W_g: jnp.ndarray, W_o: jnp.ndarray) -> jnp.ndarray:
    """
    Computes Retention using the recurrent formulation with JAX lax.scan.
    Includes K scaling and Swish gating.
    """
    seq_len, embed_dim = Q.shape
    scaling = embed_dim**-0.5
    K_scaled = K * scaling # Apply K scaling

    # Prepare inputs for scan: stack Q, K_scaled, V along a new axis
    inputs = jnp.stack([Q, K_scaled, V], axis=1) # Shape: (seq_len, 3, embed_dim)

    def scan_body(carry_S, x_t):
        """
        Args:
            carry_S: The state S from the previous step (embed_dim, embed_dim).
            x_t: The input tuple for the current step (q_n, k_n_scaled, v_n).
                 Each element has shape (embed_dim,).
        Returns:
            (new_S, ret_n): Tuple of the updated state and the output for this step.
        """
        q_n, k_n_scaled, v_n = x_t[0], x_t[1], x_t[2]
        # Update state: S_n = gamma * S_{n-1} + K_n^T V_n
        new_S = jnp.exp(gamma) * carry_S + jnp.outer(k_n_scaled, v_n)
        # Calculate output: Retention(X_n) = Q_n S_n
        ret_n = q_n @ new_S
        return new_S, ret_n

    # Initial state
    S_initial = jnp.zeros((embed_dim, embed_dim), dtype=jnp.float32)

    # Run the scan
    _, retention_result = scan(scan_body, S_initial, inputs)

    # Apply Swish Gating
    gated_output = (swish(K @ W_g) * retention_result) @ W_o
    return gated_output


def chunkwise_retention_refactored(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray, gamma: float, chunk_size: int, W_g: jnp.ndarray, W_o: jnp.ndarray) -> jnp.ndarray:
    """
    Computes Retention using a refactored chunkwise formulation with JAX.
    Includes K scaling and Swish gating. Uses lax.scan for state accumulation.
    """
    seq_len, embed_dim = Q.shape
    scaling = embed_dim**-0.5
    if seq_len % chunk_size != 0:
        raise ValueError("Sequence length must be divisible by chunk_size")

    num_chunks = seq_len // chunk_size

    # --- Precomputation ---
    D = create_decay_matrix(gamma, chunk_size)
    xi_vec = jnp.exp(gamma*(jnp.arange(chunk_size, dtype=jnp.float32) + 1)) # (chunk_size,)
    zeta_vec = jnp.exp(gamma*(chunk_size - 1 - jnp.arange(chunk_size, dtype=jnp.float32))) # (chunk_size,)
    gamma_B = jnp.exp(gamma*chunk_size)

    K_scaled = K * scaling # Apply K scaling

    # Reshape Q, K, V into chunks
    Q_c = Q.reshape(num_chunks, chunk_size, embed_dim)
    K_c_scaled = K_scaled.reshape(num_chunks, chunk_size, embed_dim)
    V_c = V.reshape(num_chunks, chunk_size, embed_dim)

    # Precompute Inner-Chunk results for all chunks
    inner_qk_all = (Q_c @ K_c_scaled.transpose(0, 2, 1)) * D[None, :, :]
    inner_output_all = inner_qk_all @ V_c

    # Precompute the K^T (V * zeta) term for all chunks
    v_zeta_c = V_c * zeta_vec[None, :, None]
    kv_zeta_all = K_c_scaled.transpose(0, 2, 1) @ v_zeta_c # Shape: (num_chunks, embed_dim, embed_dim)

    # --- Recurrent State Accumulation with Scan ---
    def scan_body_chunkwise(carry_R, x_kv_zeta):
        """
        Args:
            carry_R: The state R from the previous chunk (embed_dim, embed_dim).
            x_kv_zeta: The precomputed kv_zeta term for the current chunk.
        Returns:
            (new_R, prev_R): Tuple of (updated state for next iteration, state *before* update for output).
        """
        prev_R = carry_R # State before update (R_{i-1})
        # Update R state: R_i = gamma^B * R_{i-1} + K_i^T (V_i * zeta_i)
        new_R = gamma_B * prev_R + x_kv_zeta
        return new_R, prev_R

    # Initial state
    R_initial = jnp.zeros((embed_dim, embed_dim), dtype=jnp.float32)

    # Run scan over the precomputed kv_zeta terms for each chunk
    # `scan` returns the final state and the history of the *second* return value (prev_R)
    _, R_history_array = scan(scan_body_chunkwise, R_initial, kv_zeta_all) # Shape: (num_chunks, embed_dim, embed_dim)

    # --- Post-Scan Combination ---
    # Calculate Cross-Chunk results using R_history_array
    cross_qr_all = Q_c @ R_history_array
    cross_output_all = cross_qr_all * xi_vec[None, :, None]

    # Combine Inner and Cross results
    retention_result_c = inner_output_all + cross_output_all

    # Reshape back
    retention_result = retention_result_c.reshape(seq_len, embed_dim)

    # Apply Swish Gating
    gated_output = (swish(K @ W_g) * retention_result) @ W_o
    return gated_output

# --- Simulation Parameters ---
SEQ_LEN = 128
EMBED_DIM = 128
CHUNK_SIZE = 8
GAMMA = jnp.log(0.99)

# --- Generate Dummy Data & Weights with JAX PRNG ---
key = random.PRNGKey(0) # Master key
keys = random.split(key, 5) # Split key for each random generation

Q_data = random.normal(keys[0], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
K_data = random.normal(keys[1], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
V_data = random.normal(keys[2], (SEQ_LEN, EMBED_DIM), dtype=jnp.float32)
# Scaled initialization like in paper
W_g_data = random.normal(keys[3], (EMBED_DIM, EMBED_DIM), dtype=jnp.float32) * (1 / EMBED_DIM)
W_o_data = random.normal(keys[4], (EMBED_DIM, EMBED_DIM), dtype=jnp.float32) * (1 / EMBED_DIM)

# --- Run JIT-compiled Formulations ---
# Compile the functions using jax.jit for performance
recurrent_retention_jit = jax.jit(recurrent_retention)
# Mark chunk_size (argument index 4) as static
chunkwise_retention_refactored_jit = jax.jit(chunkwise_retention_refactored, static_argnums=(4,))

print("Running Recurrent Retention (JAX JIT)...")
# Block until computation is done to ensure accurate comparison
recurrent_output = recurrent_retention_jit(Q_data, K_data, V_data, GAMMA, W_g_data, W_o_data).block_until_ready()

print("Running Chunkwise Retention (Refactored JAX JIT)...")
# Block until computation is done
chunkwise_refactored_output = chunkwise_retention_refactored_jit(Q_data, K_data, V_data, GAMMA, CHUNK_SIZE, W_g_data, W_o_data).block_until_ready()

# --- Compare Results ---
print("\nComparing JAX outputs...")
print(f"Recurrent Output Shape: {recurrent_output.shape}")
print(f"Chunkwise Refactored Output Shape: {chunkwise_refactored_output.shape}")

are_close = jnp.allclose(recurrent_output, chunkwise_refactored_output)

print(f"\nJAX Outputs are equivalent (Recurrent vs. Refactored Chunkwise): {are_close}")

if not are_close:
    print("Difference between JAX outputs:")
    diff = recurrent_output - chunkwise_refactored_output
    max_diff = jnp.max(jnp.abs(diff))
    print(f"Maximum absolute difference: {max_diff}")
    mean_diff = jnp.mean(jnp.abs(diff))
    print(f"Mean absolute difference: {mean_diff}") 