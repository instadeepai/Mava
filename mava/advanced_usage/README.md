# Advanced Mava usage 👽
## Data recording from a PPO system 🔴
We include here an example of an advanced use case with Mava: recording experience data from a PPO system, which can then be used for offline MARL—e.g. using the [OG-MARL](https://github.com/instadeepai/og-marl) framework. This functionality is demonstrated in [ff_ippo_store_experience.py](./ff_ippo_store_experience.py), and uses [Flashbax](https://github.com/instadeepai/flashbax)'s `Vault` feature. Vault enables efficient storage of experience data recorded in JAX-based systems, and integrates tightly with Mava and the rest of InstaDeep's MARL ecosystem.

Firstly, a vault must be created using the structure of an experience buffer. Here, we create a dummy structure of the data we want to store:
```py
# Transition structure
dummy_flashbax_transition = {
    "done": jnp.zeros((config["system"]["num_agents"],), dtype=bool),
    "action": jnp.zeros((config["system"]["num_agents"],), dtype=jnp.int32),
    "reward": jnp.zeros((config["system"]["num_agents"],), dtype=jnp.float32),
    "observation": jnp.zeros(
        (
            config["system"]["num_agents"],
            env.observation_spec().agents_view.shape[1],
        ),
        dtype=jnp.float32,
    ),
    "legal_action_mask": jnp.zeros(
        (
            config["system"]["num_agents"],
            config["system"]["num_actions"],
        ),
        dtype=bool,
    ),
}

# Flashbax buffer
buffer = fbx.make_flat_buffer(
    max_length=int(5e6),
    min_length=int(1),
    sample_batch_size=1,
    add_sequences=True,
    add_batch_size=(
        n_devices
        * config["system"]["num_updates_per_eval"]
        * config["system"]["update_batch_size"]
        * config["arch"]["num_envs"]
    ),
)

# Buffer state
buffer_state = buffer.init(
    dummy_flashbax_transition,
)
```

We can now create a `Vault` for our data:
```py
v = Vault(
    vault_name="our_system_name",
    init_fbx_state=buffer_state,
    vault_uid="unique_vault_id",
)
```

We modify our `learn` function to additionally record our agents' trajectories, such that we can access experience data:
```py
learner_output, experience_to_store = learn(learner_state)
```

Because of the Anakin architecture set-up, our trajectories are stored in the incorrect dimensions for our use case. Hence, we transform the data, and then store it in a flashbax buffer:
```py
# Shape legend:
# D: Number of devices
# NU: Number of updates per evaluation
# UB: Update batch size
# T: Time steps per rollout
# NE: Number of environments

@jax.jit
def _reshape_experience(experience: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
    """Reshape experience to match buffer."""

    # Swap the T and NE axes (D, NU, UB, T, NE, ...) -> (D, NU, UB, NE, T, ...)
    experience: Dict[str, chex.Array] = jax.tree_map(lambda x: x.swapaxes(3, 4), experience)
    # Merge 4 leading dimensions into 1. (D, NU, UB, NE, T ...) -> (D * NU * UB * NE, T, ...)
    experience: Dict[str, chex.Array] = jax.tree_map(
        lambda x: x.reshape(-1, *x.shape[4:]), experience
    )
    return experience

flashbax_transition = _reshape_experience(
    {
        # (D, NU, UB, T, NE, ...)
        "done": experience_to_store.done,
        "action": experience_to_store.action,
        "reward": experience_to_store.reward,
        "observation": experience_to_store.obs.agents_view,
        "legal_action_mask": experience_to_store.obs.action_mask,
    }
)
# Add to fbx buffer
buffer_state = buffer.add(buffer_state, flashbax_transition)
```

Then, periodically, we can write this buffer state into the vault, which is stored on disk:
```py
v.write(buffer_state)
```

If we now want to use the recorded data, we can easily restore the vault in another context:
```py
v = Vault(
    vault_name="our_system_name",
    vault_uid="unique_vault_id",
)
buffer_state = v.read()
```

For a demonstration of offline MARL training, see some examples [here](https://github.com/instadeepai/og-marl/tree/feat/vault).

---
⚠️ Note: this functionality is highly experimental! The current API likely to change.
