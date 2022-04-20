import jax

from typing import Optional

from gym import Space

from mava.utils.debugging.environments.jax.simple_spread import Scenario
from mava.utils.debugging.environments.jax_debug_env import MultiAgentJaxEnv


scenario_name = "simple_spread"
action_space = "discrete"
num_agents = 2

scenario = Scenario()
key = jax.random.PRNGKey(42)
# if seed:
#     scenario.seed(seed)

# create world
world = scenario.make_world(num_agents, key)

# create multiagent environment
env = MultiAgentJaxEnv(
    world,
    action_space,
    reset_callback=scenario.reset_world,
    reward_callback=scenario.reward,
    observation_callback=scenario.observation,
    done_callback=scenario.done,
)


jitted_reset = jax.jit(env.reset)
# jitted_step = jax.jit(env.step)
jitted_reset(world)
# jitted_step(env, world, {"agent_0": 0, "agent_1": 0})
# env.step(world, {"agent_0": 0, "agent_1": 0})

