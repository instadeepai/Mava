# Mava with end to end JAX training

## How to setup
Inside of the `jax_distribution` folder, run:

### Python virtual env
- Using venv:
    ```
    make venv
    ```
- Using conda:
    ```
    conda create -n mava_env python=3.9
    conda activate mava_env
    pip install --upgrade pip setuptools
    pip install -r requirements.txt
    ```


## When using accelerators
For different accelerators it is required to install the correct version of JAX for that accelerator after following the above instructions. Please run the appropriate command in either your python virtual environment or conda environment.

### When using a TPU
```
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### When using a GPU
Please verify your CUDA version and run one of the following corresponding commands.

```
# CUDA 12 installation
# Note: wheels only available on linux.
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For more information and more detailed JAX installation instructions, please see the official [JAX repo](https://github.com/google/jax).
