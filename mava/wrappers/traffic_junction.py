from typing import Dict, Tuple

import dm_env
import numpy as np

from mava.utils.debugging.environments.traffic_junction import TrafficJunctionEnv
from mava.wrappers import PettingZooParallelEnvWrapper


class TrafficJunctionWrapper(PettingZooParallelEnvWrapper):
    """Environment wrapper for Traffic Junction MARL environment."""

    def __init__(self, environment: TrafficJunctionEnv):
        super().__init__(environment=environment)

        self._step_type: dm_env.StepType  # For tracking step type during step
        self._reset_next_step = False  # Whether environment should reset before next step

        # Env specs
        self.environment.action_spaces = {}
        self.environment.observation_spaces = {}
        self.environment._extras_specs = {}

        self.agent_ids = []
        for a_i in range(environment.num_agents):
            agent_id = "agent_" + str(a_i)
            self.agent_ids.append(agent_id)
            self.environment.possible_agents = self.agent_ids

        for agent_id in self.agent_ids:
            self.environment.action_spaces[agent_id] = environment.action_space
            self.environment.observation_spaces[agent_id] = environment.observation_space

        self.reset()

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[dm_env.TimeStep, np.ndarray]:
        """Steps the environment."""
        if self._reset_next_step:
            self._reset_next_step = False
            self.reset()

        observations, rewards, episode_over, extras = self._environment.step(actions)
        dones = {agent_id: False for agent_id in self.agent_ids}  # No agents are ever fully done
        communication_graph = extras['env_graph']

        if observations:
            observations = self._convert_observations(observations, dones)

        if episode_over:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        return (
            dm_env.TimeStep(
                observation=observations,
                reward=rewards,
                discount=self._discounts,
                step_type=self._step_type,
            ),
            communication_graph,
        )

    def reset(self) -> Tuple[dm_env.TimeStep, np.ndarray]:
        """Resets the episode."""
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST
