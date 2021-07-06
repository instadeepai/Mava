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

"""A simple multi-agent-system-environment training loop."""

import time
from typing import Any, Dict, Optional, Tuple

import acme
import dm_env
import numpy as np
from acme.utils import counting, loggers

import mava
from mava.types import Action
from mava.utils.wrapper_utils import (
    SeqTimestepDict,
    convert_seq_timestep_and_actions_to_parallel,
    generate_zeros_from_spec,
)


class SequentialEnvironmentLoop(acme.core.Worker):
    """A Sequential MARL environment loop.
    This takes `Environment` and `Executor` instances and coordinates their
    interaction. Executors are updated if `should_update=True`. This can be used as:
        loop = EnvironmentLoop(environment, executor)
        loop.run(num_episodes)
    A `Counter` instance can optionally be given in order to maintain counts
    between different Mava components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.
    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger from acme. A string `label` can be passed
    to easily change the label associated with the default logger; this is ignored
    if a `Logger` instance is given.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        executor: mava.core.Executor,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        should_update: bool = True,
        label: str = "sequential_environment_loop",
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._executor = executor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)
        self._should_update = should_update
        self._running_statistics: Dict[str, float] = {}
        self.num_agents = self._environment.num_agents

        # keeps track of previous actions and timesteps
        self._prev_action: Dict[str, Action] = {
            a: None for a in self._environment.possible_agents
        }
        self._prev_timestep: Dict[str, dm_env.TimeStep] = {
            a: None for a in self._environment.possible_agents
        }
        self._agent_action_timestep: Dict[str, Tuple[Action, dm_env.TimeStep]] = {}
        self._step_type: Dict[str, dm_env.StepType] = {
            a: dm_env.StepType.FIRST for a in self._environment.possible_agents
        }

        # For evaluation, this keeps track of the total undiscounted reward
        # for each agent accumulated during the episode.
        self.rewards: Dict[str, float] = {}
        self.episode_returns: Dict[str, float] = {}
        for agent, spec in self._environment.reward_spec().items():
            self.rewards.update({agent: generate_zeros_from_spec(spec)})
            self.episode_returns.update({agent: generate_zeros_from_spec(spec)})

    def _get_action(self, agent_id: str, timestep: dm_env.TimeStep) -> Any:
        return self._executor.select_action(agent_id, timestep.observation)

    def _get_running_stats(self) -> Dict:
        return self._running_statistics

    def _compute_step_statistics(self, rewards: Dict[str, float]) -> None:
        pass

    def _compute_episode_statistics(
        self,
        episode_returns: Dict[str, float],
        episode_steps: int,
        start_time: float,
    ) -> None:
        pass

    def _set_step_type(
        self, timestep: dm_env.TimeStep, step_type: dm_env.StepType
    ) -> dm_env.TimeStep:
        return dm_env.TimeStep(
            observation=timestep.observation,
            reward=timestep.reward,
            discount=timestep.discount,
            step_type=step_type,
        )

    def _send_observation(self) -> None:
        if len(self._agent_action_timestep) == self.num_agents:
            timesteps: Dict[str, SeqTimestepDict] = {
                k: {"timestep": v[1], "action": v[0]}
                for k, v in self._agent_action_timestep.items()
            }
            (
                parallel_actions,
                parallel_timestep,
            ) = convert_seq_timestep_and_actions_to_parallel(
                timesteps, self._environment.possible_agents
            )

            if parallel_timestep.step_type.first():
                assert all([val is None for val in parallel_actions.values()])
                self._executor.observe_first(parallel_timestep)
            else:
                self._executor.observe(
                    parallel_actions, next_timestep=parallel_timestep
                )

            self.rewards = parallel_timestep.reward
            for agent, reward in self.rewards.items():
                self.episode_returns[agent] = self.episode_returns[agent] + reward
            self._agent_action_timestep = {}

    def _perform_turn(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        # current agent
        agent = self._environment.current_agent

        # save action, timestep pairs for current agent
        timestep = self._set_step_type(timestep, self._step_type[agent])
        self._agent_action_timestep[agent] = (self._prev_action[agent], timestep)

        self._prev_timestep[agent] = timestep

        # obtain action given the timestep of current agent
        action = self._get_action(agent, timestep)

        # perform environment step; a new agent becomes the current agent and its
        # timestep is returned
        timestep = self._environment.step(action)

        # save the action of the former agent
        self._prev_action[agent] = action
        self._step_type[agent] = dm_env.StepType.MID

        # send observation to executor if (action, timestep) pairs are saved for all
        # agents
        # if true, executor observes the data
        self._send_observation()

        return timestep

    def _collect_last_timesteps(self, timestep: dm_env.TimeStep) -> None:
        assert timestep.step_type == dm_env.StepType.LAST
        cache_tsp = [timestep]

        self._agent_action_timestep = {}

        for _ in range(self.num_agents):
            agent = self._environment.current_agent
            timestep = self._set_step_type(timestep, dm_env.StepType.LAST)
            self._agent_action_timestep[agent] = (self._prev_action[agent], timestep)

            timestep = self._environment.step(
                generate_zeros_from_spec(self._environment.action_spec()[agent])
            )
            cache_tsp += [timestep]

        assert len(self._agent_action_timestep) == self.num_agents

        self._send_observation()

    def run_episode(self) -> loggers.LoggingData:
        """Run one episode.
        Each episode is a loop which interacts first with the environment to get an
        observation and then give that observation to the agent in order to retrieve
        an action.
        Returns:
            An instance of `loggers.LoggingData`.
        """
        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0

        self._prev_action = {a: None for a in self._environment.possible_agents}
        self._prev_timestep = {a: None for a in self._environment.possible_agents}
        self._agent_action_timestep = {}
        self._step_type = {
            a: dm_env.StepType.FIRST for a in self._environment.possible_agents
        }

        self.rewards = {}
        self.episode_returns = {}
        for agent, spec in self._environment.reward_spec().items():
            self.rewards.update({agent: generate_zeros_from_spec(spec)})
            self.episode_returns.update({agent: generate_zeros_from_spec(spec)})

        timestep = self._environment.reset()

        # Run an episode.
        while not timestep.last():
            timestep = self._perform_turn(timestep)

            # if Last timestep is encounterd, Env is frozen and observations for all
            # agents are collected. (action, timestep) pairs must be obtained for all
            # agents
            if timestep.last():
                self._collect_last_timesteps(timestep)

            # Update all actors
            if self._should_update:
                self._executor.update()

            # Book-keeping.
            episode_steps += 1

        self._compute_episode_statistics(
            self.episode_returns,
            episode_steps,
            start_time,
        )
        if self._get_running_stats():
            return self._get_running_stats()
        else:
            # Record counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)

            # Collect the results and combine with counts.
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                "episode_length": episode_steps,
                "mean_episode_return": np.mean(list(self.episode_returns.values())),
                "steps_per_second": steps_per_second,
            }
            result.update(counts)

            return result

    def run(
        self, num_episodes: Optional[int] = None, num_steps: Optional[int] = None
    ) -> None:
        """Perform the run loop.
        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None.
        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely.
        Args:
            num_episodes: number of episodes to run the loop for.
            num_steps: minimal number of steps to run the loop for.
        Raises:
            ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """

        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        episode_count, step_count = 0, 0
        while not should_terminate(episode_count, step_count):
            result = self.run_episode()
            episode_count += 1
            step_count += result["episode_length"]
            # Log the given results.
            self._logger.write(result)


class ParallelEnvironmentLoop(acme.core.Worker):
    """A parallel MARL environment loop.
    This takes `Environment` and `Executor` instances and coordinates their
    interaction. Executors are updated if `should_update=True`. This can be used as:
        loop = EnvironmentLoop(environment, executor)
        loop.run(num_episodes)
    A `Counter` instance can optionally be given in order to maintain counts
    between different Mava components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.
    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger from acme. A string `label` can be passed
    to easily change the label associated with the default logger; this is ignored
    if a `Logger` instance is given.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        executor: mava.core.Executor,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        should_update: bool = True,
        label: str = "parallel_environment_loop",
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._executor = executor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)
        self._should_update = should_update
        self._running_statistics: Dict[str, float] = {}

    def _get_actions(self, timestep: dm_env.TimeStep) -> Any:
        return self._executor.select_actions(timestep.observation)

    def _get_running_stats(self) -> Dict:
        return self._running_statistics

    def _compute_step_statistics(self, rewards: Dict[str, float]) -> None:
        pass

    def _compute_episode_statistics(
        self,
        episode_returns: Dict[str, float],
        episode_steps: int,
        start_time: float,
    ) -> None:
        pass

    def run_episode(self) -> loggers.LoggingData:
        """Run one episode.
        Each episode is a loop which interacts first with the environment to get a
        dictionary of observations and then give those observations to the executor
        in order to retrieve an action for each agent in the system.
        Returns:
            An instance of `loggers.LoggingData`.
        """

        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0

        timestep = self._environment.reset()

        if type(timestep) == tuple:
            timestep, env_extras = timestep
        else:
            env_extras = {}

        # Make the first observation.
        self._executor.observe_first(timestep, extras=env_extras)

        # For evaluation, this keeps track of the total undiscounted reward
        # for each agent accumulated during the episode.
        rewards: Dict[str, float] = {}
        episode_returns: Dict[str, float] = {}
        for agent, spec in self._environment.reward_spec().items():
            rewards.update({agent: generate_zeros_from_spec(spec)})
            episode_returns.update({agent: generate_zeros_from_spec(spec)})

        # Run an episode.
        while not timestep.last():

            # Generate an action from the agent's policy and step the environment.
            actions = self._get_actions(timestep)

            if type(actions) == tuple:
                # Return other action information
                # e.g. the policy information.
                env_actions, _ = actions
            else:
                env_actions = actions

            timestep = self._environment.step(env_actions)

            if type(timestep) == tuple:
                timestep, env_extras = timestep
            else:
                env_extras = {}

            rewards = timestep.reward

            # Have the agent observe the timestep and let the actor update itself.
            self._executor.observe(
                actions, next_timestep=timestep, next_extras=env_extras
            )

            if self._should_update:
                self._executor.update()

            # Book-keeping.
            episode_steps += 1

            # If env returns empty dict at end of episode.
            if not rewards:
                rewards = {
                    agent: generate_zeros_from_spec(spec)
                    for agent, spec in self._environment.reward_spec().items()
                }

            self._compute_step_statistics(rewards)

            for agent, reward in rewards.items():
                episode_returns[agent] = episode_returns[agent] + reward

        self._compute_episode_statistics(
            episode_returns,
            episode_steps,
            start_time,
        )
        if self._get_running_stats():
            return self._get_running_stats()
        else:
            # Record counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)

            # Collect the results and combine with counts.
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                "episode_length": episode_steps,
                "mean_episode_return": np.mean(list(episode_returns.values())),
                "steps_per_second": steps_per_second,
            }
            result.update(counts)

            return result

    def run(
        self, num_episodes: Optional[int] = None, num_steps: Optional[int] = None
    ) -> None:
        """Perform the run loop.
        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None.
        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely.
        Args:
            num_episodes: number of episodes to run the loop for.
            num_steps: minimal number of steps to run the loop for.
        Raises:
            ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """

        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        episode_count, step_count = 0, 0
        while not should_terminate(episode_count, step_count):
            result = self.run_episode()
            episode_count += 1
            step_count += result["episode_length"]
            # Log the given results.
            self._logger.write(result)
