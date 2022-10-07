from mava.components import training

# The components that a needed for IPPO with the Huber value loss.
huber_value_loss_components = [
    training.HuberValueLossFunction,
]
