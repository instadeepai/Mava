from typing import Dict, Tuple, Union, Any

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec

from mava.types import OLT
from mava.utils.environments.traffic_junction import TrafficJunctionEnv
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers import PettingZooParallelEnvWrapper


class TrafficJunctionWrapper(PettingZooParallelEnvWrapper):
    """Environment wrapper for Traffic Junction MARL environment."""

    _step_type: dm_env.StepType  # For tracking step type during step
    _discounts: Dict[str, Union[int, float]]  # Agent discounts

    def __init__(self, environment: TrafficJunctionEnv, max_steps: int = 20):
        super().__init__(environment=environment)

        self._reset_next_step = (
            False  # Whether environment should reset before next step
        )
        self.max_steps = max_steps  # Max number of steps in an episode before resetting

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
            self.environment.observation_spaces[
                agent_id
            ] = environment.observation_space

        # Compute discounts
        self._discounts = {
            agent: np.array(0, dtype='float32')
            for agent in self._environment.possible_agents
        }

        self.n_episodes = 0  # Track number of episodes in env
        self.n_steps = 0  # Track number of steps in episode

        # Reset the env
        obs, graph = self.reset()
        print("\n\n\n\n\nOBS", obs)

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[dm_env.TimeStep, Dict[str, Any]]:
        """Steps the environment."""
        if self._reset_next_step:
            self._reset_next_step = False
            self.reset()

        actions = self._action_array_from_dict(actions)
        observations, rewards, episode_over, extras = self._environment.step(actions)

        # End the episode if max steps are reached
        self.n_steps += 1
        if self.n_steps >= self.max_steps:
            episode_over = True

        dones = {
            agent_id: False for agent_id in self.agent_ids
        }  # No agents are ever fully done

        # Convert obs
        observations = self._convert_observations(observations, dones)

        # Convert rewards
        rewards = self._agent_dict_from_array(rewards)

        if episode_over:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
            discount = {
                agent: np.array(0, dtype='float32')
                for agent in self._environment.possible_agents
            }
        else:
            self._step_type = dm_env.StepType.MID
            discount = self._discounts

        return (
            dm_env.TimeStep(
                observation=observations,
                reward=rewards,
                discount=discount,
                step_type=self._step_type,
            ),
            {'communication_graph': extras['env_graph']},
        )

    def reset(self) -> Tuple[dm_env.TimeStep, Dict[str, Any]]:
        """Resets the episode."""
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        self.n_episodes += 1
        self.n_steps = 0

        observations, communication_graph = self.environment.reset(
            epoch=self.n_episodes
        )
        observations = self._convert_observations(
            observations, {agent_id: False for agent_id in self.agent_ids}
        )

        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }

        return (
            parameterized_restart(rewards, self._discounts, observations),
            {'communication_graph': communication_graph},
        )

    def _action_array_from_dict(self, action_dict: Dict):
        return np.array([action_dict[agent] for agent in self.agent_ids])

    # Convert Debugging environment observation so it's dm_env compatible.
    # Also, the list of legal actions must be converted to a legal actions mask.
    def _convert_observations(
        self, observes: np.array, dones: Dict[str, bool]
    ) -> Dict[str, OLT]:
        self.observes = self._agent_dict_from_array(observes)
        observations: Dict[str, OLT] = {}
        for agent, observation in observes.items():
            observations[agent] = OLT(
                observation=observation,
                legal_actions=np.ones(2),
                terminal=np.asarray([dones[agent]], dtype=np.float32),
            )

        return observations

    def _agent_dict_from_array(self, array):
        return {
            agent_id: array[self.agent_ids.index(agent_id)]
            for agent_id in self.agent_ids
        }

    def observation_spec(self) -> Dict[str, OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observations = self._environment._get_obs()
        return self._convert_observations(observes=observations, dones=[False] * self.environment.num_agents)

    def extra_spec(self) -> Dict[str, np.array]:
        return {'communication_graph': self.environment._get_env_graph()}

    def action_spec(
        self,
    ) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            spec for actions.
        """
        return {agent_id: np.array(1, dtype=int) for agent_id in self.agent_ids}

    def reward_spec(self) -> Dict[str, np.array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        return {agent_id: np.array(1, dtype='float32') for agent_id in self.agent_ids}

    def discount_spec(self) -> Dict[str, np.array]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        return {agent_id: np.array(1, dtype='float32') for agent_id in self.agent_ids}
