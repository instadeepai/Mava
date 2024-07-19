# Detailed installation guide

### Conda virtual environment
We recommend using `conda` for package management. These instructions should allow you to install and run mava.

1. Create and activate a virtual environment
```bash
conda create -n mava python=3.12
conda activate mava
```

2. Clone mava
```bash
git clone https://github.com/instadeepai/Mava.git
cd mava
```

3. Install the dependencies
```bash
pip install -e .
```

4. Install jax on your accelerator. The example below is for an NVIDIA GPU, please the [official install guide](https://github.com/google/jax#installation) for other accelerators
```bash
pip install "jax[cuda12]==0.4.25"
```

5. Run a system!
```bash
python mava/systems/ppo/ff_ippo.py env=rware
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

    Alternatively, run bash inside a docker container with mava installed by running `make bash`, and from there systems can be run as follows: `python dir/to/system.py`.
