# Troubleshooting and Frequently Asked Questions

## Troubleshooting

| Error                                                        | Resolution                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory` | `export LD_LIBRARY_PATH=[ENV_PATH]/lib/` , where `[ENV_PATH]` is where your python virtual environment is located. |

## Frequently Asked Questions