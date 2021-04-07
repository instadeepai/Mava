from mava import core
from acme import types
from typing import Dict
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