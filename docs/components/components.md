# Mava Components

> 🚧 **Note:** This only applies to the callback redesign of Mava.

## Callbacks
A callback function is one that is passed as an input to another function, and its execution is postponed until the function to which it is passed is run.

In the MAVA system, callbacks are invoked via core components (such as `on_execution_init`) or store variables (such as `trainer.store.step_fn`).

## Hooks mixin
Mixin class used to call system component hooks. In addition to the system abstract components, the core components inherit from the mixin classes too, as the mixin classes supply the call for component hooks.

```python
def on_execution_init(self) -> None:
    """Executor initialisation."""
    for callback in self.callbacks:
        callback.on_execution_init(self)
```

## Components

The callback design centers around combining various component classes. Each component will overwrite relevant hooks which will be executed to form a system's functionality. Each component class has a related component config class where its associated hyperparameter defaults may be defined. The hyperparameters that are defined in a component config class can also be overwritten at build time by passing in updated values to the `.build` method of a given system.
