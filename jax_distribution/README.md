# The beginning of Mava 3.0

## How to setup
Inside of the `jax_distribution` folder, run:

### Python virtual env
- venv:
    ```
    make build_30
    ```
- OR conda:
    ```
    conda create -n mava_30 python=3.9
    conda activate mava_30
    pip install --upgrade pip setuptools
    pip install -r requirements.txt
    ```

## When using accelerators
For different accelerators you would need to install the correct version of Jax for that accelerator after following the above instructions. Please run the following command in either your python virtual environment or conda environment.

### When using a TPU
```
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### When using a GPU
Please verify your CUDA version and run the corresponding command.

```
# CUDA 12 installation
# Note: wheels only available on linux.
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For more detailed installation instructions, please see the official [JAX repo](https://github.com/google/jax).
