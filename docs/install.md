# Detailed installation guide

### Docker

#### Building the image

1. Build the docker image using the `make` command:

    To build the docker image run

    ```bash
    make build
    ```

2. Run a system:

    ```bash
    make run example=dir/to/system.py
    ```

    For example, `make run example=mava/systems/ff_ippo_rware.py`.

    Alternatively, run bash inside a docker container with mava installed by running `make bash`, and from there systems can be run as follows: `python dir/to/system.py`.

### Python virtual environment

1. If not using docker, we strongly recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies in order to avoid version conflicts. To install Mava in the virtual environment, run:

    ```bash
    python3 -m venv mava
    source mava/bin/activate
    pip install --upgrade pip setuptools
    git clone https://github.com/instadeepai/Mava.git
    cd mava
    pip install .
    ```

    **If you plan on using `JAX` along with a GPU or TPU:**

    Please follow the instruction on the official [`JAX`](https://github.com/google/jax) project repository.

2. Developing features for Mava

    When developing features for Mava please add `-e` to the `pip install` and include our development requirements as follows:

    ```bash
    pip install -e .[dev]
    ```

3. Run an system:

    Inside your python virtual environment run:
    ```
    python dir/to/system.py
    ```

### Conda virtual environment
If you prefer using `conda` for package management, the instructions from the [Python virtual environment](#python-virtual-environment) section you can do the following:

```bash
conda create -n mava python=3.9
git clone https://github.com/instadeepai/Mava.git
cd mava
```
and then follow the same instructions as before.
