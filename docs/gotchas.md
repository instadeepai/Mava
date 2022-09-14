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
## Installing from source
- In order to edit the MAVA source code, you must ensure that you have cloned the repo and installed from source using `pip install -e .`
- A common mistake is to `run pip install id-mava`, which installs MAVA from pypi, resulting in a duplicate version of MAVA being stored as a dependancy.
- If you encounter `ModuleNotFoundError: No module named 'mava'` or Mava is executed by the example systems via site-packages (mava is stored as a package), a likely fix is to add the home directroy of the cloned Mava source code to `PYTHONPATH` with `export PYTHONPATH=<mava_homedir>`. On a linux-based, this command can be added to `.bashrc`, to avoid having to export the path every time a new terminal is opened.
