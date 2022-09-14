# Gotchas

> 🚧 **Note:** This only applies to the callback redesign of Mava.

## Debugging
While working on creating your own component, or your own system, you may encounter some errors that come from an unknown source. In order to help with locating and correcting issues, please see here some tricks for debugging in MAVA.

### pdb
`pdb` is an interactive source code debugger for Python programs. It helps by freezing the system at a predetermined moment and tracing all local and global variables at that point with `pdb.set_trace()`.

> 🖊️ If you try to print and use some commands before running the system and your print statements seem to be getting ignored, please ensure that Mava was installed from source using `pip install -e .`.

### Tracing the store variables
In order to see the variables that are included in one of the node stores or the builder store, one needs to use `.__dict__` which shows all attributes and values within a store instance. In order to know the values currently in the builder store for example, one may call `builder.store.__dict__.keys()`.

### Jit
Certain jax features will lead to print outputs not being displayed. For example, if one wants to print out the `trajectory` or `sample` variables from the [step function](https://github.com/instadeepai/Mava/blob/7b11a082ba790e1b2c2f0acd633ff605fffbe768/mava/components/jax/training/step.py), the output that is produced will look something like `Traced<ShapedArray(float32[5,20])>with<DynamicJaxprTrace(level=0/1)>`.
In order to have access to the underlying jax.numpy arrays that are in these variables, one can comment out the `@jit` decorators throughout the code or use the following debugging command:
```python
with jax.disable_jit():
    pdb.set_trace()
```

## Components
When creating components, it is important to include variable types for both the config class and component class `__init__` method. This is necessary because Mava uses it to determine which config variables are available in the system. In the example below, we provide the types for the config variables and also the config class variable inside the `Distributor` component. For the config class variable we use the `DistributorConfig` as the type, which points the system to the config class that is used.

```python
@dataclass
class DistributorConfig:
    num_executors: int = 1
    multi_process: bool = True
    nodes_on_gpu: Union[List[str], str] = "trainer"
    run_evaluator: bool = True
    distributor_name: str = "System"
    terminal: str = "current_terminal"
    single_process_max_episodes: Optional[int] = None
    is_test: Optional[bool] = False


class Distributor(Component):
    def __init__(self, config: DistributorConfig = DistributorConfig()):
```
