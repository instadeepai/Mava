# Wrappers

This section contains common wrappers that are used throughout Mava.

# Getting a new environment to work with Mava

Mava uses wrappers to allow different environments to conform to Mava's environment API. An example of the MPE environment creation setup can be found in [debugging_utils](https://github.com/instadeepai/Mava/blob/develop/mava/utils/environments/debugging_utils.py#L53). Here the environment is first initialised as normal. Then a wrapper gets applied to that environment that conforms to Deepmind's `dm_env.Environment` standard. Mava also requires 3 extra methods that can be found in [env_wrappers](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/env_wrappers.py).

Our wrapper for the PettingZoo environment provides a good starting example on how the wrapping is implemented. For the parallel agent action case, see the [PettingZooParallelEnvWrapper](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/pettingzoo.py#L356). Otherwise for sequential action environments, see the [PettingZooAECEnvWrapper](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/pettingzoo.py#L38).
