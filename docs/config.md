# The config

<hr>
**Note:** This only applies to the callback redesign of Mava.
<hr>

The `config` is an attribute that exists in various components; it is the container of the configuration of the components that needs to be adjusted at their initialization (e.g the learning rate for the model updating).

The `config` must be a `dataclass`, since it is an attribute to store data. Each config class has default attributes values and each component that has a `config`, must be associated with a static method called `config_class()` that returns the config class that belong to that component.

## The config restrictions
If a user want to make a new component that needs a config or to change an existing config class, there are some constraints that must be met:

- The `config` is a `dataclass`.
- All the `config` dataclass names must be unique, and cannot be used as a name of another config regardless of whether it is for a different component.
- All the `config` parameters' name must be unique. Which means if an attribute is defined in one component config, the other components config can't use that name for their attributes.


## The config handler
To maintain all the existing configurations and ensure that the necessary constraints are fulfilled, we have a class named `Config`:

::: mava.systems.jax.config
