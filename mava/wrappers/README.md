# Wrappers

This section contains common wrappers that are used throughout Mava.

## Supported Environments

A given multi-agent system interacts with its environment via an `EnvironmentLoop`. This loop takes as input a `system` instance and a multi-agent `environment`
instance which implements the [DeepMind Environment API][dm_env]. Mava currently supports multi-agent environment loops and environment wrappers for the following environments and environment suites:

* [PettingZoo][pettingzoo]
* [SMAC][smac]
* [Flatland][flatland]
* [2D RoboCup][robocup]
* [OpenSpiel][openspiel]
* [Melting pot][meltingpot]

|<img  src="../../docs/images/multiw_animation.gif" width="300px"/> | <img src="../../docs/images/sc2_animation.gif" width="200px"/>  | <img src="../../docs/images/flatland.gif" width="200px" />  |
|:---:|:---:|:---:|
|MAD4PG on PettingZoo's Multi-Walker environment. | VDN on the SMAC 3m map.| MADQN on Flatland. |

|<img  src="../../docs/images/robocup_animation.gif" width="300px"/> |<img  src="../../docs/images/madqn_meltingpot_cleanup_scenario.gif" width="300px"/> |
|:---:|:---:|
|MAD4PG on the 2D RoboCup environment using 6 executors.| MADQN on a melting pot clean up scenario |

# Getting a new environment to work with Mava

Mava uses wrappers to allow different environments to conform to Mava's environment API. For example, the following file shows how our debugging environments are made [debugging_utils](https://github.com/instadeepai/Mava/blob/develop/mava/utils/environments/debugging_utils.py#L53). Here the environment is first initialised as normal. Then a wrapper is applied to that environment that conforms to Deepmind's `dm_env.Environment` standard. Mava also requires three extra methods that can be found in [env_wrappers](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/env_wrappers.py).

Our wrapper for the PettingZoo environment provides a good starting example on how our wrappers are implemented. For an environment with a parallel environment loop (where agents perform actions simultaneously), please see the [PettingZooParallelEnvWrapper](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/pettingzoo.py#L356) as a useful starting example. Otherwise for an environment with a sequential environment loop (where agents perform actions in turn), please see the [PettingZooAECEnvWrapper](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/pettingzoo.py#L38).

[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[smac]: https://github.com/oxwhirl/smac
[openspiel]: https://github.com/deepmind/open_spiel
[meltingpot]: https://github.com/deepmind/meltingpot
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[robocup]: https://github.com/rcsoccersim
[dm_env]: https://github.com/deepmind/dm_env
