import time
import jax
from jax import random
import jax.numpy as jnp

# Function for the operation
def do_useless_comp(x, unused):
    
    x = jnp.power(x, 2)
    x = jnp.sqrt(x)    

    return x, unused

def do_useless_loop(x):
    x, _ = jax.lax.scan(
                do_useless_comp, x, None, 100
            )

    return x

# Create random data
x = random.normal(random.PRNGKey(0), (10000, 10000))

# Test on CPU
time_start = time.time()
jax.devices('cpu')[0]
y_cpu = do_useless_loop(x).block_until_ready()
cpu_time = time.time() - time_start
print(f'CPU time: {cpu_time} seconds')

# Test on GPU
if len(jax.devices())==1:
    time_start = time.time()
    jax.devices('gpu')[0]
    y_gpu = do_useless_loop(x).block_until_ready()
    gpu_time = time.time() - time_start
    print(f'GPU time: {gpu_time} seconds')
else:
    print("GPU not available")

# Test on TPU
if len(jax.devices())>1:
    time_start = time.time()
    jax.devices('tpu')[0]
    y_tpu = do_useless_loop(x).block_until_ready()
    tpu_time = time.time() - time_start
    print(f'TPU time: {tpu_time} seconds')
else:
    print("TPU not available")
