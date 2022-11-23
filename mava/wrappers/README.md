# Wrappers

This section contains common wrappers that are used throughout Mava.

## Supported Environments

A given multi-agent system interacts with its environment via an `EnvironmentLoop`. This loop takes as input a `system` instance and a multi-agent `environment`
instance which implements the [DeepMind Environment API][dm_env]. Mava currently supports multi-agent environment loops and environment wrappers for the following environments and environment suites:

* [PettingZoo][pettingzoo]
* [SMAC][smac]
* [Flatland][flatland]

|<img  src="../../docs/images/multiw_animation.gif" width="400px"/> | <img src="../../docs/images/sc2_animation.gif" width="250px"/>| <img src="../../docs/images/flatland.gif" width="300px" />  |
|:---:|:---:|:---:|
|MAD4PG on PettingZoo's Multi-Walker environment. | VDN on the SMAC 3m map.|MADQN on Flatland. |





# Getting a new environment to work with Mava

Mava uses wrappers to allow different environments to conform to Mava's environment API. For example, the following file shows how our debugging environments are made [debugging_utils](https://github.com/instadeepai/Mava/blob/develop/mava/utils/environments/debugging_utils.py#L53). Here the environment is first initialised as normal. Then a wrapper is applied to that environment that conforms to Deepmind's `dm_env.Environment` standard. Mava also requires three extra methods that can be found in [env_wrappers](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/env_wrappers.py).

Our wrapper for the PettingZoo environment provides a good starting example on how our wrappers are implemented. For an environment with a parallel environment loop (where agents perform actions simultaneously), please see the [PettingZooParallelEnvWrapper](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/pettingzoo.py#L356) as a useful starting example.

[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[smac]: https://github.com/oxwhirl/smac
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[dm_env]: https://github.com/deepmind/dm_env
