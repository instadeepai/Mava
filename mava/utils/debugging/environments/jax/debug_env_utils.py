from typing import Tuple

from jax.random import PRNGKey

from mava.utils.debugging.environments.jax.core import JaxWorld
from mava.utils.debugging.environments.jax.debug_env import MAJaxDiscreteDebugEnv
from mava.utils.debugging.environments.jax.debug_env_base import MultiAgentJaxEnvBase
from mava.utils.debugging.environments.jax.simple_spread import Scenario, make_world


def make_environment(
    num_agents: int, key: PRNGKey, action_space="discrete"
) -> MultiAgentJaxEnvBase:
    if action_space != "discrete":
        raise NotImplementedError(
            "Action spaces other than 'discrete' are not implemented in jax"
        )

    scenario = Scenario(make_world(num_agents, key))

    # create world
    world: JaxWorld = scenario.make_world(num_agents)

    # create multiagent environment
    return MAJaxDiscreteDebugEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        done_callback=scenario.done,
    )
