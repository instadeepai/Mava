# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Iterator, List, Sequence, Union

import dm_env
import numpy as np
from acme import specs, types
from acme.specs import EnvironmentSpec
from acme.testing.fakes import Actor as ActorMock
from acme.testing.fakes import ContinuousEnvironment, DiscreteEnvironment
from acme.testing.fakes import Environment as MockedEnvironment
from acme.testing.fakes import _generate_from_spec, _validate_spec

from mava import core
from mava.types import OLT, Observation
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper, SequentialEnvWrapper
from tests.system import System

"""Mock Objects for Tests"""


class MockedExecutor(ActorMock, core.Executor):
    """Mock Exexutor Class."""

    def __init__(self, spec: specs.EnvironmentSpec):
        super().__init__(spec)
        self._specs = spec

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        return {
            agent: _generate_from_spec(self._spec[agent].actions)
            for agent, observation in observations.items()
        }

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> Union[float, int]:
        return _generate_from_spec(self._spec[agent].actions)

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        for agent, observation_spec in self._specs.items():
            _validate_spec(
                observation_spec.observations,
                timestep.observation[agent],
            )
        if extras:
            _validate_spec(extras)

    def agent_observe_first(self, agent: str, timestep: dm_env.TimeStep) -> None:
        _validate_spec(self._spec[agent].observations, timestep.observation)

    def observe(
        self,
        action: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:

        for agent, observation_spec in self._spec.items():
            if agent in action.keys():
                _validate_spec(observation_spec.actions, action[agent])

            if agent in next_timestep.reward.keys():
                _validate_spec(observation_spec.rewards, next_timestep.reward[agent])

            if agent in next_timestep.discount.keys():
                _validate_spec(
                    observation_spec.discounts, next_timestep.discount[agent]
                )

            if next_timestep.observation and agent in next_timestep.observation.keys():
                _validate_spec(
                    observation_spec.observations, next_timestep.observation[agent]
                )
        if next_extras:
            _validate_spec(next_extras)

    def agent_observe(
        self,
        agent: str,
        action: Union[float, int, types.NestedArray],
        next_timestep: dm_env.TimeStep,
    ) -> None:
        observation_spec = self._spec[agent]
        _validate_spec(observation_spec.actions, action)
        _validate_spec(observation_spec.rewards, next_timestep.reward)
        _validate_spec(observation_spec.discounts, next_timestep.discount)


class MockedSystem(MockedExecutor, System):
    """Mocked System Class."""

    def __init__(
        self,
        specs: specs.EnvironmentSpec,
    ):
        super().__init__(specs)
        self._specs = specs

        # Initialize Mock Vars
        self.variables: Dict = {}
        network_type = "mlp"
        self.variables[network_type] = {}
        for agent in self._specs.keys():
            self.variables[network_type][agent] = np.random.rand(5, 5)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        variables: Dict = {}
        for network_type in names:
            variables[network_type] = {
                agent: self.variables[network_type][agent] for agent in self.agents
            }
        return variables


"""Function returns a Multi-agent env, of type base_class.
base_class: DiscreteEnvironment or ContinuousEnvironment. """


def get_ma_environment(
    base_class: Union[DiscreteEnvironment, ContinuousEnvironment]
) -> Any:
    class MockedMAEnvironment(base_class):  # type: ignore
        """Mocked Multi-Agent Environment.
        This simply creates multiple agents, with a spec per agent
        and updates the spec functions of base_class."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            base_class.__init__(self, *args, **kwargs)
            self._agents = ["agent_0", "agent_1", "agent_2"]
            self._possible_agents = self.agents
            self.num_agents = len(self.agents)

            multi_agent_specs = {}
            for agent in self.agents:
                spec = self._spec
                actions = spec.actions
                rewards = spec.rewards
                discounts = spec.discounts

                # Observation spec needs to be an OLT
                ma_observation_spec = self.observation_spec()
                multi_agent_specs[agent] = EnvironmentSpec(
                    observations=ma_observation_spec,
                    actions=actions,
                    rewards=rewards,
                    discounts=discounts,
                )

            self._specs = multi_agent_specs

        def extra_spec(self) -> Dict:
            return {}

        def reward_spec(self) -> Dict[str, specs.Array]:
            reward_specs = {}
            for agent in self.agents:
                reward_specs[agent] = super().reward_spec()
            return reward_specs

        def discount_spec(self) -> Dict[str, specs.BoundedArray]:
            discount_specs = {}
            for agent in self.agents:
                discount_specs[agent] = super().discount_spec()
            return discount_specs

        @property
        def agents(self) -> List:
            return self._agents

        @property
        def possible_agents(self) -> List:
            return self._possible_agents

        @property
        def env_done(self) -> bool:
            return not self.agents

    return MockedMAEnvironment


"""Class that updates functions for sequential environment.
This class should be inherited with a MockedMAEnvironment. """


class SequentialEnvironment(MockedEnvironment, SequentialEnvWrapper):
    def __init__(self, agents: List, specs: EnvironmentSpec) -> None:
        self._agents = agents
        self._possible_agents = agents
        self._specs = specs
        self.agent_step_counter = 0

    def agent_iter(self, n_agents: int) -> Iterator[str]:
        return iter(self.agents)

    @property
    def current_agent(self) -> Any:
        return self.agent_selection

    @property
    def agent_selection(self) -> str:
        return self.possible_agents[self.agent_step_counter]

    def observation_spec(self) -> OLT:

        if hasattr(self, "agent_selection"):
            active_agent = self.agent_selection
        else:
            active_agent = self.agents[0]
        return OLT(
            observation=super().observation_spec(),
            legal_actions=self.action_spec()[active_agent],
            terminal=specs.Array(
                (1,),
                np.float32,
            ),
        )

    def _generate_fake_reward(self) -> types.NestedArray:
        return _generate_from_spec(self._specs[self.agent_selection].rewards)

    def _generate_fake_discount(self) -> types.NestedArray:
        return _generate_from_spec(self._specs[self.agent_selection].discounts)

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        action_specs = {}
        for agent in self.agents:
            action_specs[agent] = super().action_spec()
        return action_specs

    def reset(self) -> dm_env.TimeStep:
        observation = self._generate_fake_observation()
        discount = convert_np_type("float32", 1)  # Not used in pettingzoo
        reward = convert_np_type("float32", 0)
        self._step = 1
        return parameterized_restart(
            reward=reward, discount=discount, observation=observation
        )

    def _generate_fake_observation(self) -> OLT:
        return _generate_from_spec(self.observation_spec())

    def step(self, action: Union[float, int, types.NestedArray]) -> dm_env.TimeStep:
        # Return a reset timestep if we haven't touched the environment yet.
        if not self._step:
            return self.reset()

        _validate_spec(self._spec.actions, action)

        observation = self._generate_fake_observation()
        reward = self._generate_fake_reward()
        discount = self._generate_fake_discount()

        self.agent_step_counter += 1

        if self._episode_length and (self._step == self._episode_length):
            # Only reset step once all all agents have taken their turn.
            if self.agent_step_counter == len(self.agents):
                self._step = 0
                self.agent_step_counter = 0

            # We can't use dm_env.termination directly because then the discount
            # wouldn't necessarily conform to the spec (if eg. we want float32).
            return dm_env.TimeStep(dm_env.StepType.LAST, reward, discount, observation)
        else:
            # Only update step counter once all agents have taken their turn.
            if self.agent_step_counter == len(self.agents):
                self._step += 1
                self.agent_step_counter = 0

            return dm_env.transition(
                reward=reward, observation=observation, discount=discount
            )


"""Class that updates functions for parallel environment.
This class should be inherited with a MockedMAEnvironment. """


class ParallelEnvironment(MockedEnvironment, ParallelEnvWrapper):
    def __init__(self, agents: List, specs: EnvironmentSpec) -> None:
        self._agents = agents
        self._specs = specs

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        action_spec = {}
        for agent in self.agents:
            action_spec[agent] = super().action_spec()
        return action_spec

    def observation_spec(self) -> Observation:
        observation_specs = {}
        for agent in self.agents:
            legals = self.action_spec()[agent]
            terminal = specs.Array(
                (1,),
                np.float32,
            )

            observation_specs[agent] = OLT(
                observation=super().observation_spec(),
                legal_actions=legals,
                terminal=terminal,
            )
        return observation_specs

    def _generate_fake_observation(self) -> Observation:
        return _generate_from_spec(self.observation_spec())

    def reset(self) -> dm_env.TimeStep:
        observations = {}
        for agent in self.agents:
            observation = self._generate_fake_observation()
            observations[agent] = observation

        rewards = {agent: convert_np_type("float32", 0) for agent in self.agents}
        discounts = {agent: convert_np_type("float32", 1) for agent in self.agents}

        self._step = 1
        return parameterized_restart(rewards, discounts, observations)  # type: ignore

    def step(
        self, actions: Dict[str, Union[float, int, types.NestedArray]]
    ) -> dm_env.TimeStep:

        # Return a reset timestep if we haven't touched the environment yet.
        if not self._step:
            return self.reset()

        for agent, action in actions.items():
            _validate_spec(self._specs[agent].actions, action)

        observation = {
            agent: self._generate_fake_observation() for agent in self.agents
        }
        reward = {agent: self._generate_fake_reward() for agent in self.agents}
        discount = {agent: self._generate_fake_discount() for agent in self.agents}

        if self._episode_length and (self._step == self._episode_length):
            self._step = 0
            # We can't use dm_env.termination directly because then the discount
            # wouldn't necessarily conform to the spec (if eg. we want float32).
            return dm_env.TimeStep(dm_env.StepType.LAST, reward, discount, observation)
        else:
            self._step += 1
            return dm_env.transition(
                reward=reward, observation=observation, discount=discount
            )


"""Mocked Multi-Agent Discrete Environment"""


DiscreteMAEnvironment = get_ma_environment(DiscreteEnvironment)
ContinuousMAEnvironment = get_ma_environment(ContinuousEnvironment)


class MockedMADiscreteEnvironment(
    DiscreteMAEnvironment, DiscreteEnvironment  # type: ignore
):
    def __init__(self, *args: Any, **kwargs: Any):
        DiscreteMAEnvironment.__init__(self, *args, **kwargs)


"""Mocked Multi-Agent Continuous Environment"""


class MockedMAContinuousEnvironment(
    ContinuousMAEnvironment, ContinuousEnvironment  # type: ignore
):
    def __init__(self, *args: Any, **kwargs: Any):
        ContinuousMAEnvironment.__init__(self, *args, **kwargs)


"""Mocked Multi-Agent Parallel Discrete Environment"""


class ParallelMADiscreteEnvironment(ParallelEnvironment, MockedMADiscreteEnvironment):
    def __init__(self, *args: Any, **kwargs: Any):
        MockedMADiscreteEnvironment.__init__(self, *args, **kwargs)
        ParallelEnvironment.__init__(self, self.agents, self._specs)


"""Mocked Multi-Agent Sequential Discrete Environment"""


class SequentialMADiscreteEnvironment(
    SequentialEnvironment, MockedMADiscreteEnvironment
):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        MockedMADiscreteEnvironment.__init__(self, *args, **kwargs)
        SequentialEnvironment.__init__(self, self.agents, self._specs)


"""Mocked Multi-Agent Parallel Continuous Environment"""


class ParallelMAContinuousEnvironment(
    ParallelEnvironment, MockedMAContinuousEnvironment
):
    def __init__(self, *args: Any, **kwargs: Any):
        MockedMAContinuousEnvironment.__init__(self, *args, **kwargs)
        ParallelEnvironment.__init__(self, self.agents, self._specs)


"""Mocked Multi-Agent Sequential Continuous Environment"""


class SequentialMAContinuousEnvironment(
    SequentialEnvironment, MockedMAContinuousEnvironment
):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        MockedMAContinuousEnvironment.__init__(self, *args, **kwargs)
        SequentialEnvironment.__init__(self, self.agents, self._specs)
