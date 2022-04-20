import jax

from typing import Optional

from gym import Space

from mava.utils.debugging.environments.jax.simple_spread import Scenario, make_world
from mava.utils.debugging.environments.jax_debug_env import MultiAgentJaxEnv


scenario_name = "simple_spread"
action_space = "discrete"
num_agents = 2

key = jax.random.PRNGKey(42)
scenario = Scenario(make_world(2, key))
# if seed:
#     scenario.seed(seed)

# create world
world = scenario.make_world(num_agents)

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
jitted_step = jax.jit(env.step)
jitted_reset(world)
# jitted_step(world, {"agent_0": 0, "agent_1": 0})

