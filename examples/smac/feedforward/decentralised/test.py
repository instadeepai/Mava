from smac.env import StarCraft2Env
from mava.wrappers import SMACWrapper
from mava.wrappers.env_preprocess_wrappers import ConcatAgentIdToObservation

import numpy as np

env = StarCraft2Env(map_name="3m")

env = SMACWrapper(env)

env = ConcatAgentIdToObservation(env)

spec = env.action_spec()
spec = env.observation_spec()

res = env.reset()

actions = {"agent_0": 1, "agent_1": 1, "agent_2": 1}

res = env.step(actions)

print("Done")