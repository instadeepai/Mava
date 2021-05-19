<img src="docs/images/mava_name.png" width="80%">

# Mava: a research framework for multi-agent reinforcement learning

**[Overview](#overview)** | **[Installation](#installation)** | **[Systems]** | **[Examples]** |

![PyPI Python Version](https://img.shields.io/pypi/pyversions/id-mava)
![PyPI version](https://badge.fury.io/py/id-mava.svg)
![pytest](https://github.com/arnupretorius/mava/workflows/format_and_test/badge.svg)

Mava is a library for building multi-agent reinforcement learning (MARL) systems. Mava builds off of Acme and in a similar way strives to expose simple, efficient, and readable components, as well as examples that serve both as reference implementations of popular algorithms and as strong
baselines, while still providing enough flexibility to do novel research.


## Overview

If you just want to get started using Mava quickly, the main thing to know about
the library is that we expose a number of system implementations and an
`EnvironmentLoop` primitive similar to Acme that can be used as follows:

```python
loop = mava.EnvironmentLoop(environment, system)
loop.run()
```

This will run a simple loop in which the given multi-agent system interacts with its
environment and and each agent learns from this interaction. This assumes a `system` instance
(implementations of which you can find [here][Systems]) and a multi-agent `environment`
instance which implements the [DeepMind Environment API][dm_env].

For a deeper dive, take a look at the detailed working code
examples found in our [examples] subdirectory which show how to instantiate a few MARL systems and environments.

> :information_source: Mava heavily relies on Acme, therefore as is the case with Acme, we make the same statement regarding reliability: mava is a framework for MARL research written by
> researchers, for researchers. We will make every attempt to keep everything in good
> working order, but things may break occasionally. If they do, we will make our best
> effort to fix them as quickly as possible!

## Installation

We have tested `mava` on Python 3.6, 3.7 and 3.8.

### Docker (**Recommended**)

1. Build the docker image using the following make command:
    ```bash
    make build
    ```

2. Run an example:
    ```bash
    make run EXAMPLE=dir/to/example/example.py
    ```
    For example, `make run EXAMPLE=examples/petting_zoo/run_decentralised_feedforward_maddpg.py`.

    Alternatively, you can also run a specific system that is defined in the `Makefile`:
    ```bash
    make run-maddpg
    ```
    Or run bash inside a docker container with mava installed, `make bash`, and from there examples can be run as follows: `python dir/to/example/example.py`.

    For viewing results through tensorboard, you can run
    ```bash
    make run-tensorboard EXAMPLE=dir/to/example/example.py
    ```
    and navigate to `http://127.0.0.1:6006/`.

### Python virtual environment

1.  If not using docker, we strongly recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies in order to avoid version conflicts:

    ```bash
    python3 -m venv mava
    source mava/bin/activate
    pip install --upgrade pip setuptools
    ```

2.  To install the core libraries (including [Reverb], our storage backend):

    ```bash
    pip install id-mava
    ```

    or install from source (from root directory):
    ```bash
    pip install .
    ```


3. To install Acme dependencies for [TensorFlow]-based Acme agents:

   ```
   pip install id-mava[reverb]
   pip install id-mava[tf]
   ```

   or from source:

   ```bash
   pip install .[reverb]
   pip install .[tf]
   ```

4. To install a few example environments (including [pettingzoo],
   [openspiel], and [flatland]):

   ```bash
   pip install id-mava[envs]
   ```

   or from source:
   ```bash
   pip install .[envs]
   ```

   NB: For flatland installation, It has to be installed separately using:
   ```bash
   pip install id-mava[flatland]
   ```
   or from source
   ```bash
   pip install .[flatland]
   ```

We also have a list of [optional installs](OPTIONAL_INSTALL.md) for extra functionality such as the use of Atari environments, environment wrappers, gpu support and agent episode recording.

## Contributing

Please read our [contributing docs](./CONTRIBUTING.md) for details on how to submit pull requests, our Contributor License Agreement and community guidelines.

## Troubleshooting and FAQs

Please read our [troubleshooting and FAQs guide](./TROUBLESHOOTING.md).

[Systems]: mava/systems/
[Examples]: examples/
[Tutorial]: https://arxiv.org
[Quickstart]: examples/quickstart.ipynb
[Documentation]: www.mava.rl
[Paper]: https://arxiv.org
[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[openspiel]: https://github.com/deepmind/open_spiel
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[dm_env]: https://github.com/deepmind/dm_env
