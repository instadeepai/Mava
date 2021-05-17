<img src="docs/images/mava_name.png" width="80%">

# Mava: a research framework for multi-agent reinforcement learning

**[Overview](#overview)** | **[Installation](#installation)** |
**[Documentation]** | **[Quickstart]** | **[Tutorial]** |
**[Systems]** | **[Examples]** | **[Paper]** |
**[Blog post]**

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
examples found in our [examples] subdirectory which show how to instantiate a
few MARL systems and environments. 

> :information_source: Mava heavily relies on Acme, therefore as is the case with Acme, we make the same statement regarding reliability: mava is a framework for MARL research written by
> researchers, for researchers. We will make every attempt to keep everything in good
> working order, but things may break occasionally. If they do, we will make our best
> effort to fix them as quickly as possible!

## Installation

We have tested `mava` on Python 3.6, 3.7 and 3.8.

### Docker

1. Build the docker image using the following make command:
    ```bash
    make build
    ```

2. Run bash inside a docker container with mava installed:
    ```bash
    make bash
    ```
From here, examples can be run using:
    ```
    python dir/to/example/example.py
    ```
Alternatively, examples can be run directly from the command line without first launching the container by using the following make command:
    ```
    make run EXAMPLE=dir/to/example/example.py
    ```


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

5. **Optional**: To install ROM files for Atari-Py using AutoROM (https://github.com/PettingZoo-Team/AutoROM).
   ```
   pip install autorom && AutoROM
   ```
   Then follow the on-screen instructions.

   You might also need to download unrar:
   ```
   sudo apt-get install unrar
   ```


6. **Optional**: To install opencv for [Supersuit](https://github.com/PettingZoo-Team/SuperSuit) environment wrappers.
    ```
    sudo apt-get update
    sudo apt-get install ffmpeg libsm6 libxext6  -y
    ```

7. **Optional**: To install [CUDA toolkit](https://docs.nvidia.com/cuda/) for NVIDIA GPU support, download [here](https://anaconda.org/anaconda/cudatoolkit). Alternatively, for anaconda users:

    ```bash
    conda install -c anaconda cudatoolkit
    ```

8. **Optional**: To log episodes in video/gif format, using the `Monitor` wrapper.
- Install `xvfb` to run a headless/fake screen and `ffmpeg` to record video.
    ```
    sudo apt-get install -y xvfb ffmpeg
    ```

- Setup fake display:
    ```
    xvfb-run -s "-screen 0 1400x900x24" bash
    ```

- Install `array2gif`, if you would like to save the episode in gif format.
    ```
    pip install .[record_episode]
    ```
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
[Blog post]: https://instadeep.com
[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[openspiel]: https://github.com/deepmind/open_spiel
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[dm_env]: https://github.com/deepmind/dm_env
