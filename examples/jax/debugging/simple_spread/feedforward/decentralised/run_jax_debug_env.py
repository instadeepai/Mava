import random
import time
from typing import Optional

import jax
from gym import Space

from mava.utils.debugging.environments.jax.core import JaxWorld, EntityId
from mava.utils.debugging.environments.jax.simple_spread import Scenario, make_world
from mava.utils.debugging.environments.jax_debug_env import MultiAgentJaxEnv

from mava.utils.debugging.make_env import make_debugging_env


scenario_name = "simple_spread"
action_space = "discrete"
num_agents = 5

# #--------------------------------------------------------------
key = jax.random.PRNGKey(42)
scenario = Scenario(make_world(num_agents, key))
# if seed:
#     scenario.seed(seed)

# create world
world: JaxWorld = scenario.make_world(num_agents)

# create multiagent environment
env = MultiAgentJaxEnv(
    world,
    action_space,
    reset_callback=scenario.reset_world,
    reward_callback=scenario.reward,
    observation_callback=scenario.observation,
    done_callback=scenario.done,
    # shared_viewer=False
)

jitted_reset = jax.jit(env.reset)
jitted_step = jax.jit(env.step)

world, *_ = jitted_reset(world)
for i in range(50):
    key, *agent_keys = jax.random.split(key, num_agents + 1)
    act = {
        EntityId(id=i, type=0): jax.random.randint(agent_keys[i], (), 0, 5)
        for i in range(num_agents)
    }
    world, *_ = jitted_step(world, act)

    env.render(world)
    time.sleep(0.1)

# env = make_debugging_env("simple_spread", "discrete", 2)
# env.reset()
# for i in range(50):
#     env.render()
#     env.step({"agent_0": random.randint(0, 5), "agent_1": random.randint(0, 5)})
#     time.sleep(0.2)
