# Wrappers

This section contains common wrappers that are used throughout Mava.

# Getting a new environment to work with Mava

Mava uses wrappers to allow different environments to conform to Mava's environment API. For example, the following file shows how our debugging environments are made [debugging_utils](https://github.com/instadeepai/Mava/blob/develop/mava/utils/environments/debugging_utils.py#L53). Here the environment is first initialised as normal. Then a wrapper is applied to that environment that conforms to Deepmind's `dm_env.Environment` standard. Mava also requires three extra methods that can be found in [env_wrappers](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/env_wrappers.py).

Our wrapper for the PettingZoo environment provides a good starting example on how our wrappers are implemented. For an environment with a parallel environment loop (where agents perform actions simultaneously), please see the [PettingZooParallelEnvWrapper](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/pettingzoo.py#L356) as a useful starting example. Otherwise for an environment with a sequential environment loop (where agents perform actions in turn), please see the [PettingZooAECEnvWrapper](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/pettingzoo.py#L38).
