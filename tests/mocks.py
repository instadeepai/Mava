from mava import core
from acme import types
from typing import Dict, Sequence, List
import dm_env
from acme import specs

from acme.testing.fakes import (
    _validate_spec,
    _generate_from_spec,
    Environment,
    Actor as ActorMock,
)
import numpy as np

from tests.conftest import EnvType
import pprint

from mava import specs as mava_specs

from mava.systems.system import System

#  class RandomActor(core.Actor):
#     """Fake actor which generates random actions and validates specs."""

#     def __init__(self, spec: specs.EnvironmentSpec):
#       self._spec = spec
#       self.num_updates = 0

#     def select_action(self, observation: open_spiel_wrapper.OLT) -> int:
#       _validate_spec(self._spec.observations, observation)
#       legals = np.array(np.nonzero(observation.legal_actions), dtype=np.int32)
#       return np.random.choice(legals[0])

#     def observe_first(self, timestep: dm_env.TimeStep):
#       _validate_spec(self._spec.observations, timestep.observation)

#     def observe(self, action: types.NestedArray,
#                 next_timestep: dm_env.TimeStep):
#       _validate_spec(self._spec.actions, action)
#       _validate_spec(self._spec.rewards, next_timestep.reward)
#       _validate_spec(self._spec.discounts, next_timestep.discount)
#       _validate_spec(self._spec.observations, next_timestep.observation)

#     def update(self, wait: bool = False):
#       self.num_updates += 1


class MockedExecutor(ActorMock, core.Executor):
    # Mock Executor class used in tests.

    def __init__(self, spec, env_type):
        super().__init__(spec)
        self.env_type = env_type
        # if self.env_type == EnvType.Parallel:
        #     self._specs = spec

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:

        return {
            agent: _generate_from_spec(self._spec.actions)
            for agent, observation in observations.items()
        }

    def observe_first(self, timestep: dm_env.TimeStep):
        if self.env_type == EnvType.Parallel:
            for agent, observation in self._spec.items():
                _validate_spec(observation.observations, timestep.observation[agent])

        elif self.env_type == EnvType.Sequential:
            _validate_spec(self._spec.observations, timestep.observations)


class MockedSystem(MockedExecutor, System):
    # Mock Executor class used in tests.

    def __init__(self, spec, env_type):
        super().__init__(spec, env_type)

    def get_variables(
        self, names: Dict[str, Sequence[str]]
    ) -> Dict[str, List[types.NestedArray]]:
        return None


# def make_fake_env() -> dm_env.Environment:
#     env_spec = specs.EnvironmentSpec(
#         observations=specs.Array(shape=(10, 5), dtype=np.float32),
#         actions=specs.DiscreteArray(num_values=3),
#         rewards=specs.Array(shape=(), dtype=np.float32),
#         discounts=specs.BoundedArray(
#             shape=(), dtype=np.float32, minimum=0.0, maximum=1.0
#         ),
#     )
#     return Environment(env_spec, episode_length=10)


def get_mocked_env_spec(environment) -> dm_env.Environment:
    return mava_specs.MAEnvironmentSpec(environment)