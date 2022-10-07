from mava.components import executing

# The components that are needed for a recurrent policy.
recurrent_policy_components = [
    executing.RecurrentExecutorSelectAction,
    executing.RecurrentExecutorObserve,
]
