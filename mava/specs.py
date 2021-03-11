"""Objects which specify the input/output spaces of an environment from the perspective
of each agent in a multi-agent environment.
This module exposes the same spec classes as `dm_env` as well as providing an
additional `EnvironmentSpec` class which collects all of the specs for a given
environment. An `EnvironmentSpec` instance can be created directly or by using
the `make_environment_spec` helper given a `dm_env.Environment` instance.
"""
from typing import Any, Dict, NamedTuple

import dm_env
from acme.specs import EnvironmentSpec


def make_ma_environment_spec(
    environment: dm_env.Environment,
) -> Dict[str, EnvironmentSpec]:
    """Returns an `EnvironmentSpec` describing values used by an environment for each agent."""
    specs = {}
    observation_specs = environment.observation_spec()
    action_specs = environment.action_spec()
    reward_specs = environment.reward_spec()
    discount_specs = environment.discount_spec()
    for agent in environment.possible_agents:
        specs[agent] = EnvironmentSpec(
            observations=observation_specs[agent],
            actions=action_specs[agent],
            rewards=reward_specs[agent],
            discounts=discount_specs[agent],
        )
    return specs
