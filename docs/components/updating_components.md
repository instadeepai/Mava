# Updating components

> 🚧 **Note:** This only applies to the callback redesign of Mava.

Components relating to the periodic updating of system wide parameters.

## Checkpointing components
The latest values of all [parameters in the store of the parameter server][param_server_store] (such as `trainer_steps`, `executor_steps`, `network_weights` & `optmiser state`) are periodically checkpointed/ saved to disk every [checkpoint_minute_interval][checkpointer_config] minutes. This allows us to save the state of an experiment so that we can resume training by restoring the checkpoint. The `experiment_path` (a configurable parameter) must point to a folder that contains the 'checkpoints' folder generated from a previous run. More information on config parameters may be found [here](../getting_started/config.md).

::: mava.components.jax.updating.checkpointer

## Parameter server components
::: mava.components.jax.updating.parameter_server

## System termination components
::: mava.components.jax.updating.terminators

[checkpointer_config]: https://github.com/instadeepai/Mava/blob/develop/mava/components/jax/updating/checkpointer.py#L32
[param_server_store]: https://github.com/instadeepai/Mava/blob/develop/mava/components/jax/updating/parameter_server.py#L110
