import numpy as np

def swish(x):
    return x * (1 / (1 + np.exp(-x)))

# create_decay_matrix_rl and create_cross_chunk_mask_rl remain the same
# (Copied here for completeness)
def create_decay_matrix_rl(gamma: float, size: int, dones_in_chunk: np.ndarray) -> np.ndarray:
    """
    Creates the intra-chunk decay matrix D, respecting done boundaries.
    Information does not flow from m to n if there's a done flag between m and n-1.
    """
    n_coords, m_coords = np.ogrid[:size, :size]
    diff = n_coords - m_coords
    D = np.exp(gamma * diff).astype(np.float32)
    mask_n_lt_m = n_coords < m_coords
    D[mask_n_lt_m] = 0.0
    done_indices = np.where(dones_in_chunk)[0]
    for k in done_indices:
        if k + 1 < size:
             D[k+1:, :k+1] = 0.0
    return D

def create_cross_chunk_mask_rl(size: int, dones_in_chunk: np.ndarray) -> np.ndarray:
    """
    Creates a mask to zero out cross-chunk contributions after a done flag within the chunk.
    Mask is 1 until the first done, then 0. Also used to mask K/V for state update.
    """
    mask = np.ones(size, dtype=np.float32)
    done_indices = np.where(dones_in_chunk)[0]
    if len(done_indices) > 0:
        first_done_idx = done_indices[0]
        if first_done_idx + 1 < size:
            mask[first_done_idx+1:] = 0.0
    return mask

# Recurrent function remains the same as the correctly working version
def recurrent_retention_rl(Q: np.ndarray, K: np.ndarray, V: np.ndarray, dones: np.ndarray, gamma: float, W_g: np.ndarray, W_o: np.ndarray) -> np.ndarray:
    """
    Computes Retention using the recurrent formulation (Equation 6)
    with K scaling, Swish gating, and handling RL episode boundaries (dones).
    Calculates o_n = Q_n S_n.
    """
    seq_len, embed_dim = Q.shape
    if dones.shape != (seq_len,):
        raise ValueError(f"Dones shape mismatch. Expected ({seq_len},), got {dones.shape}")

    scaling = embed_dim**-0.5
    S = np.zeros((embed_dim, embed_dim), dtype=np.float32)
    output_retention = []
    K_scaled = K * scaling

    for n in range(seq_len):
        if n > 0 and dones[n-1]:
            S.fill(0.0)
        q_n = Q[n]
        k_n_scaled = K_scaled[n]
        v_n = V[n]
        S_current_step = np.exp(gamma) * S + np.outer(k_n_scaled, v_n)
        ret_n = q_n @ S_current_step
        output_retention.append(ret_n)
        S = S_current_step # S is now S_n for the next iteration (where it's S_{n+1 - 1})

    retention_result = np.stack(output_retention, axis=0)
    gating_signal = K @ W_g
    gated_output = (swish(gating_signal) * retention_result) @ W_o
    return gated_output


# --- Modified Chunkwise (Refined State Update) ---
def chunkwise_retention_rl(Q: np.ndarray, K: np.ndarray, V: np.ndarray, dones: np.ndarray, gamma: float, chunk_size: int, W_g: np.ndarray, W_o: np.ndarray) -> np.ndarray:
    """
    Computes Retention using the chunkwise formulation, handling RL episode boundaries.
    Includes K scaling, Swish gating, and refined state update logic.
    """
    seq_len, embed_dim = Q.shape
    if seq_len % chunk_size != 0:
        raise ValueError("Sequence length must be divisible by chunk_size for chunkwise RL")
    if dones.shape != (seq_len,):
        raise ValueError(f"Dones shape mismatch. Expected ({seq_len},), got {dones.shape}")

    num_chunks = seq_len // chunk_size
    scaling = embed_dim**-0.5

    # --- Precomputation (Constants) ---
    xi_vec = np.exp(gamma * (np.arange(chunk_size, dtype=np.float32) + 1))
    # zeta[j] = exp(gamma * (chunk_size - 1 - j)) = gamma^(chunk_size - 1 - j)
    zeta_vec = np.exp(gamma * (chunk_size - 1 - np.arange(chunk_size, dtype=np.float32)))
    gamma_B = np.exp(gamma * chunk_size)

    K_scaled = K * scaling

    # Reshape inputs into chunks
    Q_c = Q.reshape(num_chunks, chunk_size, embed_dim)
    K_c_scaled = K_scaled.reshape(num_chunks, chunk_size, embed_dim)
    V_c = V.reshape(num_chunks, chunk_size, embed_dim)
    dones_c = dones.reshape(num_chunks, chunk_size)

    # --- Process Chunks Iteratively (Handling State and Masks) ---
    # R_in represents the state passed IN from the previous chunk's *last episode*
    R_in = np.zeros((embed_dim, embed_dim), dtype=np.float32)
    all_inner_outputs = []
    all_cross_outputs = []

    for i in range(num_chunks):
        q_i = Q_c[i]
        k_i_scaled = K_c_scaled[i]
        v_i = V_c[i]
        dones_i = dones_c[i]

        # R_in for this chunk is R_out from the previous chunk calculation
        # (It's already correctly reset to 0 if the previous chunk ended an episode)

        # 2. Calculate RL-aware Intra-Chunk Decay Matrix D_i
        D_i = create_decay_matrix_rl(gamma, chunk_size, dones_i)

        # 3. Calculate Inner-Chunk Retention
        inner_qk_i = (q_i @ k_i_scaled.T) * D_i
        inner_output_i = inner_qk_i @ v_i
        all_inner_outputs.append(inner_output_i)

        # 4. Calculate Cross-Chunk Retention (using state R_in from previous chunk)
        cross_qr_i = q_i @ R_in # Use state passed *in*
        # Mask application of cross-chunk state based on dones *within* this chunk
        cross_mask_i = create_cross_chunk_mask_rl(chunk_size, dones_i)
        cross_output_i = cross_qr_i * xi_vec[:, None] * cross_mask_i[:, None]
        all_cross_outputs.append(cross_output_i)

        # 5. Calculate the state R_out to be passed to the *next* chunk
        #    This state represents only the accumulation from the *last episode* in this chunk.
        done_indices_in_chunk = np.where(dones_i)[0]
        if len(done_indices_in_chunk) == 0:
            start_j = 0 # Episode started before or at the beginning of this chunk
        else:
            start_j = np.max(done_indices_in_chunk) + 1 # Episode started after the last done

        if start_j >= chunk_size: # Episode ended exactly at the end of the chunk or before
             kv_zeta_i_last_ep = np.zeros((embed_dim, embed_dim), dtype=np.float32)
        else:
            last_ep_mask = np.zeros(chunk_size, dtype=np.float32)
            last_ep_mask[start_j:] = 1.0
            k_last_ep = k_i_scaled * last_ep_mask[:, None]
            v_last_ep = v_i * last_ep_mask[:, None]
            # Sum K^T V zeta only over the last episode
            kv_zeta_i_last_ep = k_last_ep.T @ (v_last_ep * zeta_vec[:, None])

        # Determine R_out based on whether the last episode started in this chunk
        if start_j == 0:
            # Episode spans from previous chunk or starts at index 0 here
            # Combine incoming state (decayed) with this chunk's full contribution
            R_out = gamma_B * R_in + kv_zeta_i_last_ep # kv_zeta includes all steps as last_ep_mask is all 1s
        else:
            # Episode started within this chunk (after a done). State resets.
            # Outgoing state ONLY depends on the last episode within this chunk.
            R_out = kv_zeta_i_last_ep

        # Update R_in for the next iteration
        R_in = R_out

    # --- Combine Results ---
    inner_retention_full = np.concatenate(all_inner_outputs, axis=0)
    cross_retention_full = np.concatenate(all_cross_outputs, axis=0)
    retention_result = inner_retention_full + cross_retention_full

    # Apply Swish Gating
    gating_signal = K @ W_g # Using original K for gating signal
    gated_output = (swish(gating_signal) * retention_result) @ W_o
    return gated_output

# --- Simulation Parameters ---
SEQ_LEN = 1024
EMBED_DIM = 4
CHUNK_SIZE = 8
GAMMA = np.log(0.9)

# --- Generate Dummy Data & Weights ---
np.random.seed(42)
Q_data = np.random.randn(SEQ_LEN, EMBED_DIM).astype(np.float32)
K_data = np.random.randn(SEQ_LEN, EMBED_DIM).astype(np.float32)
V_data = np.random.randn(SEQ_LEN, EMBED_DIM).astype(np.float32)
W_g_data = np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32) * (1 / EMBED_DIM)
W_o_data = np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32) * (1 / EMBED_DIM)

# --- Generate Dummy Dones ---
dones_data = np.zeros(SEQ_LEN, dtype=bool)
dones_data[5] = True
dones_data[128] = True # End of chunk 1
dones_data[514] = True # End of chunk 2
dones_data[768] = True # End of chunk 3

print("Dones:", dones_data.astype(int))
print("-" * 20)

# --- Run Formulations ---
print("Running Recurrent Retention (RL)...")
recurrent_output_rl = recurrent_retention_rl(Q_data, K_data, V_data, dones_data, GAMMA, W_g_data, W_o_data)

print("\nRunning Chunkwise Retention (RL)...")
chunkwise_output_rl = chunkwise_retention_rl(Q_data, K_data, V_data, dones_data, GAMMA, CHUNK_SIZE, W_g_data, W_o_data)

# --- Compare Results ---
print("\nComparing RL outputs...")
print(f"Recurrent RL Output Shape: {recurrent_output_rl.shape}")
print(f"Chunkwise RL Output Shape: {chunkwise_output_rl.shape}")

are_close_rl = np.allclose(recurrent_output_rl, chunkwise_output_rl, atol=1e-6)

print(f"\nOutputs are equivalent (Recurrent RL vs. Chunkwise RL): {are_close_rl}")

if not are_close_rl:
    print("Difference between RL outputs:")
    max_diff = np.max(np.abs(recurrent_output_rl - chunkwise_output_rl))
    print(f"Maximum absolute difference: {max_diff}")
    mean_diff = np.mean(np.abs(recurrent_output_rl - chunkwise_output_rl))
    print(f"Mean absolute difference: {mean_diff}")

    diff_indices = np.where(~np.isclose(recurrent_output_rl, chunkwise_output_rl, atol=1e-6))
    print("First few differing indices (row, col):")
    for i in range(min(5, len(diff_indices[0]))):
        r, c = diff_indices[0][i], diff_indices[1][i]
        print(f"  Index ({r}, {c}): Rec={recurrent_output_rl[r, c]:.8f}, Chunk={chunkwise_output_rl[r, c]:.8f}, Diff={recurrent_output_rl[r, c] - chunkwise_output_rl[r, c]:.4e}")