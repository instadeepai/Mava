<img src="docs/logos/mava.png" width="50%">

# Mava: a research framework for multi-agent reinforcement learning

**[Overview](#overview)** | **[Installation](#installation)** |
**[Documentation]** | **[Quickstart]** | **[Tutorial]** |
**[Systems]** | **[Examples]** | **[Paper]** |
**[Blog post]**

![PyPI Python Version](https://img.shields.io/pypi/pyversions/id-mava)
![PyPI version](https://badge.fury.io/py/id-mava.svg)
![pytest](https://github.com/arnupretorius/mava/workflows/pytest/badge.svg)

Mava is a library for building multi-agent reinforcement learning (MARL) systems. Mava builds off of Acme and in a similar way strives to expose simple, efficient, and readable components, as well as examples that serve both as reference implementations of popular algorithms and as strong
baselines, while still providing enough flexibility to do novel research. 


## Overview

If you just want to get started using Acme quickly, the main thing to know about
the library is that we expose a number of agent implementations and an
`EnvironmentLoop` primitive that can be used as follows:

```python
loop = acme.EnvironmentLoop(environment, agent)
loop.run()
```

This will run a simple loop in which the given agent interacts with its
environment and learns from this interaction. This assumes an `agent` instance
(implementations of which you can find [here][Agents]) and an `environment`
instance which implements the [DeepMind Environment API][dm_env]. Each
individual agent also includes a `README.md` file describing the implementation
in more detail. Of course, these two lines of code definitely simplify the
picture. To actually get started, take a look at the detailed working code
examples found in our [examples] subdirectory which show how to instantiate a
few agents and environments. We also include a
[quickstart notebook][Quickstart].

Acme also tries to maintain this level of simplicity while either diving deeper
into the agent algorithms or by using them in more complicated settings. An
overview of Acme along with more detailed descriptions of its underlying
components can be found by referring to the [documentation]. And we also include
a [tutorial notebook][Tutorial] which describes in more detail the underlying
components behind a typical Acme agent and how these can be combined to form a
novel implementation.

> :information_source: Acme is first and foremost a framework for RL research written by
> researchers, for researchers. We use it for our own work on a daily basis. So
> with that in mind, while we will make every attempt to keep everything in good
> working order, things may break occasionally. But if so we will make our best
> effort to fix them as quickly as possible!

## Installation

We have tested `acme` on Python 3.6 & 3.7.

1.  **Optional**: We strongly recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies in order to avoid version conflicts:

    ```bash
    python3 -m venv mava
    source mava/bin/activate
    pip install --upgrade pip setuptools
    ```

1.  To install the core libraries (including [Reverb], our storage backend):

    ```bash
    pip install id-mava
    ```

1.  To install Acme dependencies for [TensorFlow]-based Acme agents:

    ```bash
    pip install dm-acme
    pip install dm-acme[reverb]
    pip install dm-acme[tf]
    ```

1.  Finally, to install a few example environments (including [pettingzoo],
    [openspiel], and [flatland]):

    ```bash
    pip install dm-mava[envs]
    ```
    

## Citing Mava

If you use Mava in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@article{pretorius2021mava,
    title={Mava: A Research Framework for Multi-Agent Reinforcement Learning},
    author={Arnu Pretorius and Kale-ab Tessera and Siphelele Danisa and Kevin Eloff 
    and Claude Formanek and St John Grimbly and Jonathan Shock and Herman Kamper 
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
