# Detailed installation guide

### uv virtual environment
We recommend using [uv](https://docs.astral.sh/uv/) for package management. These instructions should allow you to install and run mava.

1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone mava
```bash
git clone https://github.com/instadeepai/Mava.git
cd mava
```

3. Create a virtual environment and install requirements (this only installs a CPU version of JAX)
```bash
uv sync -p=3.12
```

3.1 If you want to install Mava so that it runs on your accelerator simply run the following:
```bash
uv sync --extra cuda12  # GPU aware JAX
uv sync --extra tpu  # TPU aware JAX
...
**Note:** If this does not work, please see the [official JAX install guide](https://github.com/google/jax#installation).

4. Run a system!
```bash
uv run mava/systems/ppo/anakin/ff_ippo.py env=rware
```
or
```
source .venv/bin/activate
python mava/systems/ppo/anakin/ff_ippo.py env=rware
```

5. (Optional) Alternate package manager

If you prefer installation can also be done using `pip`:
```bash
pip install -e ".[cuda12]"  # GPU aware JAX
pip install -e ".[tpu]"  # TPU aware JAX
```

### Docker

If you are having trouble with dependencies we recommend using our docker image and provided Makefile.

1. Build the docker image using the `make` command:

    To build the docker image run

    ```bash
    make build
    ```

2. Run a system:

    ```bash
    make run example=dir/to/system.py
    ```

    For example, `make run example=mava/systems/ppo/ff_ippo.py`.

    Alternatively, run bash inside a docker container with Mava installed by running `make bash`, and from there systems can be run as follows: `python dir/to/system.py`.
