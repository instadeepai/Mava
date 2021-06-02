<img src="docs/images/mava.png" width="80%">

# Mava: a research framework for distributed multi-agent reinforcement learning

**[Overview](#overview)** | **[Installation](#installation)** | **[Systems]** | **[Examples]** |

<!-- ![PyPI Python Version](https://img.shields.io/pypi/pyversions/id-mava) -->
<!-- ![PyPI version](https://badge.fury.io/py/id-mava.svg) -->
![pytest](https://github.com/arnupretorius/mava/workflows/format_and_test/badge.svg)

Mava is a library for building multi-agent reinforcement learning (MARL) systems. Mava builds off of [Acme][Acme] and in a similar way strives to expose simple, efficient, and readable components, as well as examples that serve both as reference implementations of popular algorithms and as strong
baselines, while still providing enough flexibility to do novel research.
## Overview
### Systems and the Executor-Trainer paradigm

At the core of the Mava framework is the concept of a `system`. A system refers to a full multi-agent reinforcement learning algorithm consisting of the following specific components: an `Executor`, a `Trainer` and a `Dataset`. 

<p style="text-align:center;">
<img src="docs/images/mava_system.png" width="45%">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="docs/images/mava_distributed_training.png" width="45%">
</p>

The `Executor` is the part of the system that interacts with the environment, takes actions for each agent and observes the next state as a collection of observations, one for each agent in the system. Essentially, executors are the multi-agent version of the Actor class in Acme and are themselves constructed through feeding to the executor a dictionary of policy networks. The `Trainer` is responsible for sampling data from the Dataset originally collected from the executor and updating the parameters for every agent in the system. Trainers are therefore the multi-agent version of the Learner class in Acme. The `Dataset` stores all of the information collected by the executors in the form of a collection of dictionaries for the actions, observations and rewards with keys corresponding to the individual agent ids.
Several examples of system implementations can be viewed [here][Systems].

### Distributed system training

Mava shares much of the design philosophy of Acme for the same reason: to allow a high level of composability for novel research (i.e. building new systems) as well as making it possible to scale systems in a simple way, using the same underlying multi-agent RL system code. In the latter case, the system executor (which is responsible for data collection) is distributed across multiple processes each with a copy of the environment. Each process collects and stores data which the Trainer uses to update the parameters of the actor networks used within each executor.

### Supported environments and the system-environment loop

A given multi-agent system interacts with its environment via an `EnvironmentLoop`. This loop takes as input a `system` instance and a multi-agent `environment`
instance which implements the [DeepMind Environment API][dm_env]. Mava currently supports multi-agent environment loops and environment wrappers for the following environments and environment suites: 
* [PettingZoo][pettingzoo]
* [SMAC][smac]
* [Flatland][flatland]
* [2D RoboCup][robocup] 

## Implementations

* 🟩 - Multi-Agent Deep Q-Networks (MADQN).
* 🟩 - Multi-Agent Deep Deterministic Policy Gradient (MADDPG).
* 🟩 - Multi-Agent Distributed Distributional Deep Deterministic Policy Gradient (MAD4PG).
* 🟨 - Multi-Agent Proximal Policy Optimisation (MAPPO).
* 🟥 - Value Decomposition Networks (VDN).
* 🟥 - Monotonic value function factorisation (QMIX).

## Examples

```python
# Mava imports
from mava.systems.tf import madqn
from mava.components.tf.architectures import DecentralisedPolicyActor
from mava.systems.tf.system.helpers import environment_factory, network_factory

# Launchpad imports
import launchpad

# Distributed program
program = madqn.MADQN(
        environment_factory=environment_factory,
        network_factory=network_factory,
        architecture=DecentralisedPolicyActor,
        num_executors=2,
    ).build()

# Launch
launchpad.launch(
        program,
        launchpad.LaunchType.LOCAL_MULTI_PROCESSING,
    )
```

For a deeper dive, take a look at the detailed working code
examples found in our [examples] subdirectory which show how to instantiate a few MARL systems and environments.
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
    For example, `make run EXAMPLE=examples/petting_zoo/run_decentralised_feedforward_maddpg_continous.py`. Alternatively, run bash inside a docker container with mava installed, `make bash`, and from there examples can be run as follows: `python dir/to/example/example.py`.

    To run an example with tensorboard viewing enabled, you can run
    ```bash
    make run-tensorboard EXAMPLE=dir/to/example/example.py
    ```
    and navigate to `http://127.0.0.1:6006/`.

3. Install multi-agent Starcraft 2 environment [Optional]:
    To install the environment, please run the provided bash script, which is a slightly modified version of the script found [here][pymarl].
    ```bash
    ./install_sc2.sh

### Python virtual environment

1.  If not using docker, we strongly recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies in order to avoid version conflicts. Please note that since Launchpad only supports Linux based OSes, using a python virtual environment will only work in these cases:

    ```bash
    python3 -m venv mava
    source mava/bin/activate
    pip install --upgrade pip setuptools
    ```

2.  To install the core libraries with dependencies from source (from root directory): 

    ```bash
    pip install -e ".[tf,envs,reverb,launchpad]"
    ```

    Note that the dependencies may be installed selectively by adding and removing their identifiers. Additional optional dependencies include `record_episode` for installing packages required to make video recordings of evaluation runs and `testing_formatting` for running tests and code formatting checks. Extra information on optional installs are given below.

3.  **NB**: Flatland and SMAC installations have to be done separately. Flatland can be installed using:
   
    ```bash
    pip install .[flatland]
    ```
   
    For StarCraft II installation, this must be installed separately according to your operating system.
    To install the StarCraft II ML environment and associated packages, please follow the instructions on [PySC2](https://github.com/deepmind/pysc2) to install the StarCraft II game files.
    Please ensure you have the required game maps (for both PySC2 and SMAC) extracted in the StarCraft II maps directory.
    Once this is done you can install the packages for the single agent case (PySC2) and the multi-agent case (SMAC).
   
    ```bash
    pip install pysc2
    pip install git+https://github.com/oxwhirl/smac.git
    ```

We also have a list of [optional installs](OPTIONAL_INSTALL.md) for extra functionality such as the use of Atari environments, environment wrappers, gpu support and agent episode recording.

## Debugging

Simple spread debugging environment. 

<p style="text-align:center;">
<img src="docs/images/simple_spread.png" width="30%">
</p>

## Roadmap

[![](https://api.gh-polls.com/poll/01F75ZJZXE8C5JM7MQWEX9PRXQ/Sequential%20environment%20support%20including%20OpenSpiel)](https://api.gh-polls.com/poll/01F75ZJZXE8C5JM7MQWEX9PRXQ/Sequential%20environment%20support%20including%20OpenSpiel/vote) <br /> 
[![](https://api.gh-polls.com/poll/01F75ZJZXE8C5JM7MQWEX9PRXQ/Population%20based%20training)](https://api.gh-polls.com/poll/01F75ZJZXE8C5JM7MQWEX9PRXQ/Population%20based%20training/vote) <br /> 
[![](https://api.gh-polls.com/poll/01F75ZJZXE8C5JM7MQWEX9PRXQ/Dynamic%20networked%20architectures)](https://api.gh-polls.com/poll/01F75ZJZXE8C5JM7MQWEX9PRXQ/Dynamic%20networked%20architectures/vote)

## Contributing

Please read our [contributing docs](./CONTRIBUTING.md) for details on how to submit pull requests, our Contributor License Agreement and community guidelines.

## Troubleshooting and FAQs

Please read our [troubleshooting and FAQs guide](./TROUBLESHOOTING.md).
## Citing Mava

If you use Mava in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@article{anon2021mava,
    title={Mava: A Research Framework for Distributed Multi-Agent Reinforcement Learning},
    author={Anonymous authors},
    year={2021},
    journal={arXiv preprint},
    url={},
}
```

[Acme]: https://github.com/deepmind/acme
[Systems]: mava/systems/
[Examples]: examples/
[Tutorial]: https://arxiv.org
[Quickstart]: examples/quickstart.ipynb
[Documentation]: www.mava.rl
[Paper]: https://arxiv.org
[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[smac]: https://github.com/oxwhirl/smac
[openspiel]: https://github.com/deepmind/open_spiel
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[robocup]: https://github.com/rcsoccersim/rcssserver
[dm_env]: https://github.com/deepmind/dm_env
[pymarl]: https://github.com/oxwhirl/pymarl
