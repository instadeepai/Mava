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
from acme.testing.fakes import Actor as ActorMock
from acme.testing.fakes import DiscreteEnvironment, _generate_from_spec, _validate_spec

from mava import core
from mava import specs as mava_specs
from mava.systems.system import System
from mava.utils.wrapper_utils import OLT, convert_np_type, parameterized_restart
from tests.conftest import EnvSpec, EnvType

"""Mock Objects for Tests"""


class MockedExecutor(ActorMock, core.Executor):
    """ Mock Exexutor Class."""

    def __init__(self, spec: specs.EnvironmentSpec):
        super().__init__(spec)

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

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        for agent, observation_spec in self._spec.items():
            _validate_spec(
                observation_spec.observations, timestep.observation[agent].observation
            )

    def agent_observe_first(self, agent: str, timestep: dm_env.TimeStep) -> None:
        _validate_spec(self._spec[agent].observations, timestep.observation.observation)

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ) -> None:

        for agent, observation_spec in self._spec.items():
            _validate_spec(observation_spec.actions, action[agent])
            _validate_spec(observation_spec.rewards, next_timestep.reward[agent])
            _validate_spec(observation_spec.discounts, next_timestep.discount[agent])
            if next_timestep.observation:
                _validate_spec(
                    observation_spec.observations, next_timestep.observation[agent]
                )

    def agent_observe(
        self,
        agent: str,
        action: Union[float, int],
        next_timestep: dm_env.TimeStep,
    ) -> None:
        observation_spec = self._spec[agent]
        _validate_spec(observation_spec.actions, action)
        # _validate_spec(observation_spec.observations, next_timestep.observation)

        # print(f"herere {observation_spec.rewards}{next_timestep.reward}")
        _validate_spec(observation_spec.rewards, next_timestep.reward)
        _validate_spec(observation_spec.discounts, next_timestep.discount)


class MockedSystem(MockedExecutor, System):
    """Mocked System Class. """

    def __init__(
        self,
        spec: specs.EnvironmentSpec,
    ):
        super().__init__(spec)
        self._spec = spec

        # Initialize Mock Vars
        self.variables: Dict = {}
        network_type = "mlp"
        self.variables[network_type] = {}
        for agent in self._spec.keys():
            self.variables[network_type][agent] = np.random.rand(5, 5)

    def get_variables(self, names: Dict[str, Sequence[str]]) -> Dict[str, List[Any]]:
        variables: Dict = {}
        for network_type in names:
            variables[network_type] = {}
            for agent in self.agents:
                variables[network_type][agent] = self.variables[network_type][agent]
        return variables


class MockedMADiscreteEnvironment(DiscreteEnvironment):
    def __init__(self, env_type: EnvType, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.agents = ["agent_0", "agent_1", "agent_2"]
        self.possible_agents = self.agents
        self.num_agents = len(self.agents)
        self.env_type = env_type

        multi_agent_specs = {}
        for agent in self.agents:
            multi_agent_specs[agent] = self._spec
        self._specs = multi_agent_specs

    def extra_spec(self) -> Dict:
        return {}

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in self.possible_agents:
            observation_specs[agent] = super().observation_spec()
        return observation_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = super().reward_spec()
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = super().discount_spec()
        return discount_specs


class SequentialMADiscreteEnvironment(MockedMADiscreteEnvironment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_selection = self.agents[0]

    def agent_iter(self, n_agents: int) -> Iterator[str]:
        return iter(self.agents)

    def _generate_fake_observation(self) -> types.NestedArray:
        return _generate_from_spec(self._specs[self.agent_selection].observations)

    def _generate_fake_reward(self) -> types.NestedArray:
        return _generate_from_spec(self._specs[self.agent_selection].rewards)

    def _generate_fake_discount(self) -> types.NestedArray:
        return _generate_from_spec(self._specs[self.agent_selection].discounts)

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        action_specs = {}
        for agent in self.possible_agents:
            action_specs[agent] = super(DiscreteEnvironment, self).action_spec()
        return action_specs

    def reset(self) -> dm_env.TimeStep:
        observation = self._generate_fake_observation()
        agent = self.agent_selection
        legals = np.ones(
            self.action_spec()[agent].shape,
            np.float32,
        )

        rewards = convert_np_type("float32", 0)
        discounts = convert_np_type("float32", 1)
        done = False
        observation = OLT(
            observation=observation,
            legal_actions=legals,
            terminal=np.asarray([done], dtype=np.float32),
        )
        self._step = 1
        return parameterized_restart(rewards, discounts, observation)


class ParallelMADiscreteEnvironment(MockedMADiscreteEnvironment):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        return {
            agent: super(DiscreteEnvironment, self).action_spec()
            for agent in self.possible_agents
        }

    def reset(self) -> dm_env.TimeStep:
        observations: Dict[str, OLT] = {}
        dones = {agent: False for agent in self.possible_agents}
        for agent in self.possible_agents:
            observation = self._generate_fake_observation()
            legals = np.ones(
                self.action_spec()[agent].shape,
                dtype=np.float32,
            )
            observations[agent] = OLT(
                observation=observation,
                legal_actions=legals,
                terminal=np.asarray([dones[agent]], dtype=np.float32),
            )
        rewards = {
            agent: convert_np_type("float32", 0) for agent in self.possible_agents
        }
        discounts = {
            agent: convert_np_type("float32", 1) for agent in self.possible_agents
        }

        self._step = 1
        return parameterized_restart(rewards, discounts, observations)

    def step(self, actions: Dict[str, Union[float, int]]) -> dm_env.TimeStep:
        # Return a reset timestep if we haven't touched the environment yet.
        if not self._step:
            return self.reset()

        for agent, action in actions.items():
            _validate_spec(self._spec.actions, action)

        observation = {
            agent: self._generate_fake_observation() for agent in self.possible_agents
        }
        reward = {agent: self._generate_fake_reward() for agent in self.possible_agents}
        discount = {
            agent: self._generate_fake_discount() for agent in self.possible_agents
        }

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


def get_mocked_env(
    env_spec: EnvSpec,
) -> Union[ParallelMADiscreteEnvironment, SequentialMADiscreteEnvironment]:
    env = None
    if "discrete" in env_spec.env_name.lower():
        if env_spec.env_type == EnvType.Parallel:
            env = ParallelMADiscreteEnvironment(
                num_actions=18,
                num_observations=2,
                obs_shape=(84, 84, 4),
                obs_dtype=np.float32,
                episode_length=10,
                env_type=env_spec.env_type,
            )
        else:
            env = SequentialMADiscreteEnvironment(
                num_actions=18,
                num_observations=2,
                obs_shape=(84, 84, 4),
                obs_dtype=np.float32,
                episode_length=10,
                env_type=env_spec.env_type,
            )
    # elif "continous" in env_spec.env_name.lower():
    #     return None
    else:
        raise Exception("Env_spec is not valid.")
    return env


def get_mocked_env_spec(environment: dm_env.Environment) -> dm_env.Environment:
    return mava_specs.MAEnvironmentSpec(environment)
