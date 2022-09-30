# The config



The `config` is an attribute that exists in various components and contains the parameter configuration to be used by components at their initialisation. Consider, for example, a parameter like `learning_rate` for the `MAPGMinibatchUpdate` component. The parameters that are defined in a component config class can also be overwritten at build time by passing in updated values to the `.build` method of a given system.
## Config restrictions
If a user wants to make a new component that needs a config or wants to change an existing config class, the following constraints must be met:

- The `config` must be a `dataclass`
- The `config` parameter names must be unique across all config dataclasses that are created. This means that if an attribute is defined in one component config, any other component config dataclasses can't use a similar parameter name.

## The config handler
To maintain all the existing configurations and ensure that the necessary constraints are fulfilled, we have a class named `Config`:

::: mava.systems.config
