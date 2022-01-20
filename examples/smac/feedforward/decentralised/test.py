from smac.env import StarCraft2Env
from mava.wrappers import SMACWrapper
from mava.wrappers.env_preprocess_wrappers import ConcatAgentIdToObservation
from mava.wrappers.env_preprocess_wrappers import ConcatPrevActionToObservation
import numpy as np

env = StarCraft2Env(map_name="3m")

env = SMACWrapper(env)

env = ConcatAgentIdToObservation(env)

env = ConcatPrevActionToObservation(env)

spec = env.action_spec()
# for agent in spec:
#     print(spec[agent].num_values)
spec = env.observation_spec()

res = env.reset()

actions = {"agent_0": 1, "agent_1": 2, "agent_2": 3}

res = env.step(actions)

print("Done")