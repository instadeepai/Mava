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


class SystemSpec:
    def __init__(self, environment: dm_env.Environment):
        self._environment = environment
        self.spec = self.make_ma_environment_spec()

    def make_ma_environment_spec(
        self,
    ) -> Dict[str, EnvironmentSpec]:
        """Returns an `EnvironmentSpec` describing values used by an environment for each agent."""
        specs = {}
        observation_specs = self._environment.observation_spec()
        action_specs = self._environment.action_spec()
        reward_specs = self._environment.reward_spec()
        discount_specs = self._environment.discount_spec()
        for agent in self._environment.possible_agents:
            specs[agent] = EnvironmentSpec(
                observations=observation_specs[agent],
                actions=action_specs[agent],
                rewards=reward_specs[agent],
                discounts=discount_specs[agent],
            )
        return specs

    def get_agent_type_spec(self) -> Dict[str, EnvironmentSpec]:
        specs = {}
        agent_types = list(set([agent.split("_")[0] for agent in self.spec.keys()]))
        for agent_type in agent_types:
            specs[agent_type] = self._spec[f"{agent_type}_0"]
        return specs

    def get_agent_types(self) -> List[str]:
        return list(set([agent.split("_")[0] for agent in self.spec.keys()]))

    def get_agent_ids(self) -> List[str]:
        return list(self.spec.keys())

    def get_agent_info(self) -> List[str]:
        return self.get_agent_ids(), self.get_agent_types()