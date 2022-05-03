import time

import jax

from mava.utils.debugging.environments.jax.core import JaxWorld, EntityId
from mava.utils.debugging.environments.jax.simple_spread import Scenario, make_world
from mava.utils.debugging.environments.jax.debug_env import MAJaxDiscreteDebugEnv
from mava.utils.debugging.environments.jax.debug_env_utils import make_environment

scenario_name = "simple_spread"
action_space = "discrete"
num_agents = 10
seed = 1
key = jax.random.PRNGKey(seed)

# #--------------------------------------------------------------
# scenario = Scenario(make_world(num_agents, key))
# # if seed:
# #     scenario.seed(seed)
#
# # create world
# world: JaxWorld = scenario.make_world(num_agents)
#
# # create multiagent environment
# env = MAJaxDiscreteDebugEnv(
#     world,
#     reset_callback=scenario.reset_world,
#     reward_callback=scenario.reward,
#     observation_callback=scenario.observation,
#     done_callback=scenario.done,
#     # shared_viewer=False
# )

env = make_environment(num_agents, key)

jitted_reset = jax.jit(env.reset)
jitted_step = jax.jit(env.step)

world, *_ = jitted_reset(key)
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
