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

# See SMAC here: https://github.com/oxwhirl/smac
# Documentation available at smac/blob/master/docs/smac.md

"""Wraps a StarCraft II MARL environment (SMAC) as a dm_env environment."""

from typing import Any, Dict, List, Optional, Tuple, Type

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

from mava.utils.environments.render_utils import Renderer

try:
    from smac.env import StarCraft2Env  # type:ignore
except ModuleNotFoundError:
    pass


from mava import types
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper


class SMACEnvWrapper(ParallelEnvWrapper):
    """
    Wraps a StarCraft II MARL environment (SMAC) as a Mava Parallel environment.
    Based on RLlib wrapper provided by SMAC.
    """

    def __init__(self, environment: StarCraft2Env) -> None:
        """Create a new multi-agent StarCraft env compatible with RLlib.
        Arguments:
            smac_args (dict): Arguments to pass to the underlying
                smac.env.starcraft.StarCraft2Env instance.
        """
        self._environment = environment

        self._reset_next_step = True
        self._possible_agents = [
            f"agent_{i}" for i in range(self._environment.get_env_info()["n_agents"])
        ]
        self._agents = self._possible_agents
        self.observation_space = {
            "observation": Box(-1, 1, shape=(self._environment.get_obs_size(),)),
            "action_mask": Box(0, 1, shape=(self._environment.get_total_actions(),)),
        }
        self.observation_spaces = {
            agent: self.observation_space["observation"] for agent in self._agents
        }
        self.action_space: Type[Discrete] = Discrete(
            self._environment.get_total_actions()
        )
        self.action_spaces = {agent: self.action_space for agent in self._agents}

        self.reward: dict = {}
        self.renderer: Optional[Renderer] = None

    def reset(self) -> Tuple[dm_env.TimeStep, np.array]:
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        self._env_done = False
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # reset internal SC2 env
        obs_list, state = self._environment.reset()

        # Convert observations
        observe: Dict[str, np.ndarray] = {}

        for i, obs in enumerate(obs_list):
            observe[f"agent_{i}"] = {
                "observation": obs,
                "action_mask": np.array(
                    self._environment.get_avail_agent_actions(i), dtype=np.float32
                ),
            }

        observations = self._convert_observations(
            observe, {agent: False for agent in self._possible_agents}
        )

        self._agents = list(observe.keys())

        # create discount spec
        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._possible_agents
        }

        # create rewards spec
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self._possible_agents
        }

        # dm_env timestep
        timestep = parameterized_restart(rewards, self._discounts, observations)

        return timestep, {"s_t": state}

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[dm_env.TimeStep, np.array]:
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        if self._reset_next_step:
            return self.reset()

        actions_feed = list(actions.values())
        reward, terminated, info = self._environment.step(actions_feed)
        obs_list = self._environment.get_obs()
        state = self._environment.get_state()
        self._env_done = terminated

        observe = {}
        rewards = {}
        dones = {}
        for i, obs in enumerate(obs_list):
            agent = f"agent_{i}"
            observe[agent] = {
                "observation": obs,
                "action_mask": np.array(
                    self._environment.get_avail_agent_actions(i), dtype=np.float32
                ),
            }
            rewards[agent] = reward
            dones[agent] = terminated

        observations = self._convert_observations(observe, dones)
        self._agents = list(observe.keys())
        rewards_spec = self.reward_spec()

        #  Handle empty rewards
        if not rewards:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, 0)
                for agent in self._possible_agents
            }
        else:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, reward)
                for agent, reward in rewards.items()
            }

        if self.env_done():
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        timestep = dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

        self.reward = rewards

        return timestep, {"s_t": state}

    def env_done(self) -> bool:
        """
        Returns a bool indicating if all agents in env are done.
        """
        return self._env_done

    def _convert_observations(
        self, observes: Dict[str, np.ndarray], dones: Dict[str, bool]
    ) -> types.Observation:
        observations: Dict[str, types.OLT] = {}
        for agent, observation in observes.items():
            if isinstance(observation, dict) and "action_mask" in observation:
                legals = observation["action_mask"]
                observation = observation["observation"]
            else:
                legals = np.ones(
                    _convert_to_spec(self.action_space).shape,
                    dtype=self.action_space.dtype,
                )
            observations[agent] = types.OLT(
                observation=observation,
                legal_actions=legals,
                terminal=np.asarray([dones[agent]], dtype=np.float32),
            )

        return observations

    def observation_spec(self) -> types.Observation:
        return {
            agent: types.OLT(
                observation=_convert_to_spec(self.observation_space["observation"]),
                legal_actions=_convert_to_spec(self.observation_space["action_mask"]),
                terminal=specs.Array((1,), np.float32),
            )
            for agent in self._possible_agents
        }

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        return {
            agent: _convert_to_spec(self.action_space)
            for agent in self._possible_agents
        }

    def reward_spec(self) -> Dict[str, specs.Array]:
        return {agent: specs.Array((), np.float32) for agent in self._possible_agents}

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        return {
            agent: specs.BoundedArray((), np.float32, minimum=0, maximum=1.0)
            for agent in self._possible_agents
        }

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        state = self._environment.get_state()
        # TODO (dries): What should the real bounds be of the state spec?
        return {
            "s_t": specs.BoundedArray(
                state.shape, np.float32, minimum=float("-inf"), maximum=float("inf")
            )
        }

    def seed(self, random_seed: int) -> None:
        self._environment._seed = random_seed
        # Reset after setting seed
        self.full_restart()

    @property
    def agents(self) -> List:
        return self._agents

    @property
    def possible_agents(self) -> List:
        return self._possible_agents

    @property
    def environment(self) -> ParallelEnv:
        """Returns the wrapped environment."""
        return self._environment

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)

    def render(self, mode: str = "human") -> Any:
        if self.renderer is None:
            self.renderer = Renderer(self, mode)
        assert mode == self.renderer.mode, "mode must be consistent across render calls"
        return self.renderer.render(mode)

    def close(self) -> None:
        """Close StarCraft II."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._environment.close()

    def full_restart(self) -> None:
        """Full restart. Closes the SC2 process and launches a new one."""
        if self._sc2_proc:
            self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1
