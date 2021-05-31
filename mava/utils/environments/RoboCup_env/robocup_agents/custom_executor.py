# type: ignore
from typing import Dict, Optional

import dm_env
from acme import types

from mava.utils.environments.RoboCup_env.robocup_utils.game_object import Flag

true_flag_coords = Flag.FLAG_COORDS


class CustomExecutor:
    """A fixed executor to test in the environments are working."""

    def __init__(self, agents):
        # Convert action and observation specs.
        self.agents = agents

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {"": ()},
    ) -> None:
        pass

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:
        pass

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        actions = {}
        for agent, observation in observations.items():
            # Pass the observation through the policy network.
            actions[agent] = self.agents[agent].get_action(observation.observation)

        # Return a numpy array with squeezed out batch dimension.
        return actions

    def update(self, wait: bool = False) -> None:
        pass
