from mava.components import executing, training

# The components that are needed for a recurrent policy.
recurrent_policy_components = [
    executing.RecurrentExecutorSelectAction,
    executing.RecurrentExecutorObserve,
]

# The components that a needed for IPPO with the Huber value loss.
huber_value_loss_components = [
    training.HuberValueLossFunction,
]
