# Gotchas

> 🚧 **Note:** This only applies to the callback redesign of Mava.

## Debugging
While working on creating your own component, or your own system, you may encounter some errors that comes from an unknown source. So, in order to locate and correct the issue, here are some tricks to debug in MAVA.

### pdb
`pdb` is an interactive source code debugger for Python programs. It helps in freezing the system at a predetermined moment and tracing all local and global variables at that point with `pdb.set_trace()`.

> 🖊️ If you try to print and use some commands before running the system and they were ignored and it didn’t work try to run `pip install -e .`.

### Tracing the store variables
To see the variables that are included in one of the nodes or the builder store you need to use `.__dict__` which shows all the attributes and their values within an instance. So in order to know the ones in the builder for example we use `builder.store.__dict__.keys()`.

### Jit
If you want to trace the `trajectory` or a `sample` for example from the [step function](https://github.com/instadeepai/Mava/blob/7b11a082ba790e1b2c2f0acd633ff605fffbe768/mava/components/jax/training/step.py), you will end up having this kind of outputs `Traced<ShapedArray(float32[5,20])>with<DynamicJaxprTrace(level=0/1)>`.
In order to have the jax.numpy arrays that are in these variables, you can comment `@jit` at the beginning of the function, or use this following commands:
```python
with jax.disable_jit():
	 pdb.set_trace()
```