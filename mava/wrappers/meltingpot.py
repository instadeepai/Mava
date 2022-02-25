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

"""Wraps a meltingpot environment to be used as a dm_env environment in mava."""
import os
from typing import Any, Callable, Dict, List, Optional, Union

import dm_env
import numpy as np
from acme import specs
from acme.types import NestedArray

from mava import types
from mava.wrappers.env_wrappers import ParallelEnvWrapper

try:
    import pygame  # type: ignore
    from meltingpot.python.scenario import Scenario  # type: ignore
    from meltingpot.python.substrate import Substrate  # type: ignore
except ModuleNotFoundError:
    Scenario = Any
    Substrate = Any


def obs_preprocessor(observation: Dict[str, NestedArray]) -> np.ndarray:
    """Converts the observation to a single array
    Meltingpot observations come as Dictionary of Arrays
    Args:
        observation (Dict[str, np.ndarray]): Observation from environment
    Returns:
        np.ndarray: Processed observation
    """
    return np.array(observation["RGB"] / 255, np.float32)


class MeltingpotEnvWrapper(ParallelEnvWrapper):
    """Environment wrapper for Melting pot."""

    def __init__(
        self,
        environment: Union[Substrate, Scenario],
        preprocessor: Callable[[Dict[str, NestedArray]], np.ndarray] = obs_preprocessor,
    ):
        """Constructor for Melting pot wrapper.
        Args:
            environment (Substrate or Scenario): Melting pot substrate or scenario.
            preprocessor (Callable[[Dict[str, NestedArray]], np.ndarray]): function that
             transforms an observation to a single array
        """
        self._environment = environment
        self._reset_next_step = True
        self._env_done = False
        self._num_agents = len(self._environment.action_spec())
        self._num_actions = self._environment.action_spec()[0].num_values
        self._env_image: Optional[np.ndarray] = None
        self._screen = None
        self._preprocessor = preprocessor

        # individual agent obervation
        _, _, _, obs = self._environment.reset()
        ob = self._preprocessor(obs[0])
        self._ob_spec = specs.Array(
            shape=ob.shape, dtype=ob.dtype, name="indv_agent_ob"
        )

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.
        Returns:
            dm_env.TimeStep: dm timestep.
        """

        timestep = self._environment.reset()
        self._reset_next_step = False

        self._set_env_image()

        return self._refine_timestep(timestep)

    def _set_env_image(self) -> None:
        """Sets an image of the environment from a timestep
        The image is from the observation key 'WORLD.RGB'
        """
        self._env_image = self._environment.observation()["WORLD.RGB"]

    def _to_olt(
        self, observation: Dict[str, NestedArray], num_values: int, is_terminal: bool
    ) -> types.OLT:
        """Createa an OLT from a observation.
        It just computes the legal actions and terminal. All actions are legal and
        terminal is determined with timestep.last()
        Args:
            observation (TimeStep): the observation
            num_values (int): the number of actions
            is_terminal (bool): whether its a terminal observation
        Returns:
            types.OLT: observation, legal actions, and terminal
        """
        legal_actions = np.ones([num_values], dtype=np.int32)
        terminal = np.asarray([is_terminal], dtype=np.float32)
        return types.OLT(
            observation=self._preprocessor(observation),
            legal_actions=legal_actions,
            terminal=terminal,
        )

    def _to_dict_observation(
        self, observation: List[Dict[str, NestedArray]], is_terminal: bool
    ) -> Dict[str, types.OLT]:
        """Observation list to dict
        Transforms a list of observations into a dictionary of observations
        with keys corresponding to the agent ids
        Args:
            observation (List[Dict[str, NestedArray]]): List observation
            is_terminal (bool): whether the observations corresponds to a
             terminal timestep
        Returns:
            Dict[str, types.OLT]: Dictionary of observations
        """
        return {
            f"agent_{i}": self._to_olt(obs, self._num_actions, is_terminal)
            for i, obs in enumerate(observation)
        }

    def _to_dict_rewards(self, rewards: List[NestedArray]) -> Dict[str, NestedArray]:
        """List of rewards to Dict of rewards
        Transforms a list of rewards to a dictionary of rewards with keys corresponding
        to the agent ids
        Args:
            rewards (List[NestedArray]): List of rewards
        Returns:
            Dict[str, NestedArray]: Dictionary of reward
        """
        return {
            f"agent_{i}": np.dtype("float32").type(rew) for i, rew in enumerate(rewards)
        }

    def _to_dict_discounts(
        self, discounts: List[NestedArray]
    ) -> Dict[str, NestedArray]:
        """List of dicounts to Dict of discounts
        Transforms a list of discounts into a dictionary of discounts with keys
        corresponding to the agent ids
        Args:
            discounts (List[NestedArray]): List of discounts
        Returns:
            Dict[str, NestedArray]: Dictionary of discounts
        """
        return {
            f"agent_{i}": np.dtype("float32").type(disc)
            for i, disc in enumerate(discounts)
        }

    def _refine_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        """Converts a melting pot timestep into one compatiple with Mava
        The difference between the timestep from melting pot and that of Mava is
        that for the observation, reward, and discount, mava expects dictionaries
        with keys corresponding to the agent ids while melting pot simply uses a
        list for this.
        Args:
            timestep (dm_env.TimeStep): a timestep from melting pot
        Returns:
            dm_env.TimeStep: a timestep compatible with Mava
        """
        is_terminal = timestep.last()
        observation = self._to_dict_observation(timestep.observation, is_terminal)
        reward = self._to_dict_rewards(timestep.reward)
        discount = self._to_dict_discounts([timestep.discount] * self._num_agents)
        return dm_env.TimeStep(timestep.step_type, reward, discount, observation)

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.
        Args:
            actions (Dict[str, np.ndarray]): actions per agent.
        Returns:
            dm_env.TimeStep: dm timestep
        """

        if self._reset_next_step:
            return self.reset()

        actions_ = [actions[f"agent_{i}"] for i in range(self._num_agents)]
        timestep = self._environment.step(actions_)
        self._set_env_image()
        timestep = self._refine_timestep(timestep)

        if timestep.last():
            self._reset_next_step = True
            self._env_done = True

        return timestep

    def render(
        self, mode: str = "human", screen_width: int = 800, screen_height: int = 600
    ) -> Optional[np.ndarray]:
        """Renders the environment in a pygame window or returns an image
        Args:
            mode (str, optional): mode for the display either rgb_array or human.
            Defaults to "human".
            screen_width (int, optional): the screen width. Defaults to 800.
            screen_height (int, optional): the screen height. Defaults to 600.
        Raises:
            ValueError: for invalid mode
        Returns:
            [np.ndarray]: an image array for mode, rgb_array
        """
        if self._env_image:
            image = self._env_image
            height, width, _ = image.shape
            scale = min(screen_height // height, screen_width // width)
            if mode == "human":
                os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
                if self._screen is None:
                    pygame.init()
                    self._screen = pygame.display.set_mode(  # type: ignore
                        (screen_width * scale, screen_height * scale)
                    )
                image = np.transpose(image, (1, 0, 2))  # PyGame is column major!
                surface = pygame.surfarray.make_surface(image)
                rect = surface.get_rect()
                surf = pygame.transform.scale(
                    surface, (rect[2] * scale, rect[3] * scale)
                )
                self._screen.blit(surf, dest=(0, 0))  # type: ignore
                pygame.display.update()
                return None
            elif mode == "rgb_array":
                return image
            else:
                raise ValueError("bad value for render mode")
        return None

    def close(self) -> None:
        """Closes the rendering screen"""
        if self._screen is not None:
            import pygame

            pygame.quit()
            self._screen = None

    def env_done(self) -> bool:
        """Check if env is done.
        Returns:
            bool: bool indicating if env is done.
        """
        done = not self._agents or self._env_done
        return done

    def observation_spec(self) -> types.Observation:
        """Observation spec.
        Returns:
            types.Observation: spec for environment.
        """
        observation_spec = self._environment.observation_spec()
        return {
            f"agent_{i}": types.OLT(
                observation=self._ob_spec,
                legal_actions=specs.Array((self._num_actions,), np.int32),
                terminal=specs.Array((1,), np.float32),
            )
            for i, spec in enumerate(observation_spec)
        }

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.
        Returns:
            Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]: spec for actions.
        """
        action_spec = self._environment.action_spec()
        return {
            f"agent_{i}": specs.DiscreteArray(spec.num_values, np.int64)
            for i, spec in enumerate(action_spec)
        }

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.
        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        return {
            f"agent_{i}": specs.Array((), np.float32) for i in range(self._num_agents)
        }

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.
        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_spec = self._environment.discount_spec()
        return {
            f"agent_{i}": specs.BoundedArray(
                (),
                np.float32,
                minimum=int(discount_spec.minimum),
                maximum=int(discount_spec.minimum),
            )
            for i in range(self._num_agents)
        }

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Extra data spec.
        Returns:
            Dict[str, specs.BoundedArray]: spec for extra data.
        """
        return {}

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).
        Returns:
            List: alive agents in env.
        """
        return [f"agent_{i}" for i in range(self._num_agents)]

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.
        Returns:
            List: all possible agents in env.
        """
        return [f"agent_{i}" for i in range(self._num_agents)]

    @property
    def environment(self) -> Union[Substrate, Scenario]:
        """Returns the wrapped environment.
        Returns:
            ParallelEnv: parallel env.
        """
        return self._environment

    @property
    def current_agent(self) -> Any:
        """Current active agent.
        Returns:
            Any: current agent.
        """
        return "agent_0"

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.
        Args:
            name (str): attribute.
        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
