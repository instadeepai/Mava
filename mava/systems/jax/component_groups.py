from mava.components.jax import executing

# The components that a needed for IPPO with a recurrent policy.
recurrent_policy_components = [
    executing.RecurrentExecutorSelectAction,
    executing.RecurrentExecutorObserve,
]