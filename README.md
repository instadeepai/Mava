<img src="docs/images/mava.png" width="20%">

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
instance which implements the [DeepMind Environment API][dm_env]. Each
individual system also includes a `README.md` file describing the implementation
in more detail.

For a deeper dive, take a look at the detailed working code
examples found in our [examples] subdirectory which show how to instantiate a
few MARL systems and environments. We also include a
[quickstart notebook][Quickstart].

> :information_source: Mava heavily relies on Acme, therefore as is the case with Acme, we make the same statement regarding reliability: mava is a framework for MARL research written by
> researchers, for researchers. We will make every attempt to keep everything in good
> working order, but things may break occasionally. If they do, we will make our best
> effort to fix them as quickly as possible!

## Installation

We have tested `mava` on Python 3.6 & 3.7.

1.  **Optional**: We strongly recommend using a
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


3.  To install Acme dependencies for [TensorFlow]-based Acme agents:

    ```bash
    pip install dm-acme
    pip install dm-acme[reverb]
    pip install dm-acme[tf]
    ```

4.  Finally, to install a few example environments (including [pettingzoo],
    [openspiel], and [flatland]):

    ```bash
    pip install id-mava[envs]
    ```

    or from source:
    ```bash
    pip install .[envs]
    ```

## Contributing

Please read our [contributing docs](./CONTRIBUTING.md) for details on how to submit pull requests, our Contributor License Agreement and community guidelines.

## Troubleshooting and FAQs

Please read our [troubleshooting and FAQs guide](./TROUBLESHOOTING.md). 

## Citing Mava

If you use Mava in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@article{pretorius2021mava,
    title={Mava: A Research Framework for Multi-Agent Reinforcement Learning},
    author={Arnu Pretorius and Kale-ab Tessera and Andries P. Smit and Siphelele Danisa and Kevin Eloff
    and Claude Formanek and St John Grimbly and Lawrence Francis and Jonathan Shock and Herman Kamper
    and Herman Engelbrecht and Alexandre Laterre and Karim Beguir},
    year={2021},
    journal={arXiv preprint},
    url={},
}
```

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
