# Troubleshooting and Frequently Asked Questions

## Troubleshooting

| Error                                                        | Resolution                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory` | `export LD_LIBRARY_PATH=[ENV_PATH]/lib/` , where `[ENV_PATH]` is where your python virtual environment is located. |
| `Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory;` | `conda install -c anaconda cudatoolkit`                      |
| GPU crashes immeditely due to a lack of memory. Add this to the top of the main file, before any other code. | import tensorflow as tf<br/>import os<br/>physical_devices = tf.config.list_physical_devices("GPU")<br/>if physical_devices:<br/>    tf.config.experimental.set_memory_growth(physical_devices[0], True)<br/>os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" |




## Frequently Asked Questions
