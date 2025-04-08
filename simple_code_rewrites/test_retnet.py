import numpy as np

def swish(x):
    return x * (1 / (1 + np.exp(-x)))

def create_decay_matrix(gamma: float, size: int) -> np.ndarray:
    """Creates the intra-chunk decay matrix D."""
    n, m = np.ogrid[:size, :size]
    D = np.where(n >= m, np.exp(gamma*(n - m)), 0)
    return D

def recurrent_retention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, gamma: float, W_g: np.ndarray, W_o: np.ndarray) -> np.ndarray:
    """
    Computes Retention using the recurrent formulation (Equation 6)
    with K scaling and Swish gating.
    """
    seq_len, embed_dim = Q.shape
    scaling = embed_dim**-0.5
    S = np.zeros((embed_dim, embed_dim), dtype=np.float32) # Ensure float32 for S
    output_retention = []

    K_scaled = K * scaling # Apply K scaling

    for n in range(seq_len):
        q_n = Q[n]
        k_n_scaled = K_scaled[n] # Use scaled K
        v_n = V[n]

        S = np.exp(gamma) * S + np.outer(k_n_scaled, v_n)
        ret_n = q_n @ S
        output_retention.append(ret_n)

    retention_result = np.stack(output_retention, axis=0)

    # Apply Swish Gating
    gated_output = (swish(K @ W_g) * retention_result) @ W_o
    return gated_output

def chunkwise_retention_refactored(Q: np.ndarray, K: np.ndarray, V: np.ndarray, gamma: float, chunk_size: int, W_g: np.ndarray, W_o: np.ndarray) -> np.ndarray:
    """
    Computes Retention using a refactored chunkwise formulation,
    separating precomputation, state accumulation, and final combination.
    Includes K scaling and Swish gating.
    """
    seq_len, embed_dim = Q.shape
    scaling = embed_dim**-0.5
    if seq_len % chunk_size != 0:
        raise ValueError("Sequence length must be divisible by chunk_size")

    num_chunks = seq_len // chunk_size

    # --- Precomputation ---
    D = create_decay_matrix(gamma, chunk_size)
    xi_vec = np.exp(gamma*(np.arange(chunk_size, dtype=np.float32) + 1)) # (chunk_size,)
    zeta_vec = np.exp(gamma*(chunk_size - 1 - np.arange(chunk_size, dtype=np.float32))) # (chunk_size,)
    gamma_B = np.exp(gamma*chunk_size)

    K_scaled = K * scaling # Apply K scaling

    # Reshape Q, K, V into chunks
    # Shape: (num_chunks, chunk_size, embed_dim)
    Q_c = Q.reshape(num_chunks, chunk_size, embed_dim)
    K_c_scaled = K_scaled.reshape(num_chunks, chunk_size, embed_dim)
    V_c = V.reshape(num_chunks, chunk_size, embed_dim)

    # Precompute Inner-Chunk results for all chunks
    # inner_qk_all shape: (num_chunks, chunk_size, chunk_size)
    inner_qk_all = (Q_c @ K_c_scaled.transpose(0, 2, 1)) * D[None, :, :]
    # inner_output_all shape: (num_chunks, chunk_size, embed_dim)
    inner_output_all = inner_qk_all @ V_c

    # Precompute the K^T (V * zeta) term needed for the recurrent state update for all chunks
    # kv_zeta_all shape: (num_chunks, embed_dim, embed_dim)
    v_zeta_c = V_c * zeta_vec[None, :, None] # Add chunk dim and embed dim for broadcast
    kv_zeta_all = K_c_scaled.transpose(0, 2, 1) @ v_zeta_c

    # --- Recurrent State Accumulation Loop ---
    R = np.zeros((embed_dim, embed_dim), dtype=np.float32)
    R_history = [] # To store the state *before* each chunk update R_{i-1}

    for i in range(num_chunks):
        R_history.append(R.copy()) # Store R_{i-1}
        # Update R state: R_i = gamma^B * R_{i-1} + K_i^T (V_i * zeta_i)
        R = gamma_B * R + kv_zeta_all[i]

    # Convert history to array: (num_chunks, embed_dim, embed_dim)
    R_history_array = np.stack(R_history, axis=0)

    # --- Post-Loop Combination ---
    # Calculate Cross-Chunk results for all chunks using Q_c and R_history
    # cross_qr_all shape: (num_chunks, chunk_size, embed_dim)
    cross_qr_all = Q_c @ R_history_array
    # cross_output_all shape: (num_chunks, chunk_size, embed_dim)
    cross_output_all = cross_qr_all * xi_vec[None, :, None] # Add chunk dim and embed dim for broadcast

    # Combine Inner and Cross results
    # retention_result_c shape: (num_chunks, chunk_size, embed_dim)
    retention_result_c = inner_output_all + cross_output_all

    # Reshape back to (seq_len, embed_dim)
    retention_result = retention_result_c.reshape(seq_len, embed_dim)

    # Apply Swish Gating (using original K)
    gated_output = (swish(K @ W_g) * retention_result) @ W_o
    return gated_output

# --- Simulation Parameters ---
SEQ_LEN = 1024
EMBED_DIM = 128
CHUNK_SIZE = 8
GAMMA = np.log(0.99)

# --- Generate Dummy Data & Weights ---
Q_data = np.random.randn(SEQ_LEN, EMBED_DIM).astype(np.float32)
K_data = np.random.randn(SEQ_LEN, EMBED_DIM).astype(np.float32)
V_data = np.random.randn(SEQ_LEN, EMBED_DIM).astype(np.float32)
W_g_data = np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32) * (1 / EMBED_DIM)
W_o_data = np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32) * (1 / EMBED_DIM)

# --- Run Formulations ---
print("Running Recurrent Retention...")
recurrent_output = recurrent_retention(Q_data, K_data, V_data, GAMMA, W_g_data, W_o_data)

print("Running Chunkwise Retention (Refactored)...")
chunkwise_refactored_output = chunkwise_retention_refactored(Q_data, K_data, V_data, GAMMA, CHUNK_SIZE, W_g_data, W_o_data)

# --- Compare Results ---
print("\nComparing outputs...")
print(f"Recurrent Output Shape: {recurrent_output.shape}")
print(f"Chunkwise Refactored Output Shape: {chunkwise_refactored_output.shape}")

are_close = np.allclose(recurrent_output, chunkwise_refactored_output)

print(f"\nOutputs are equivalent (Recurrent vs. Refactored Chunkwise): {are_close}")

if not are_close:
    print("Difference between outputs:")
    max_diff = np.max(np.abs(recurrent_output - chunkwise_refactored_output))
    print(f"Maximum absolute difference: {max_diff}")
    mean_diff = np.mean(np.abs(recurrent_output - chunkwise_refactored_output))
    print(f"Mean absolute difference: {mean_diff}")
