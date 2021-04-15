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

"""Generic environment loop wrapper to track system statistics"""

import time
from typing import Dict

import numpy as np
from acme import types
from acme.utils import loggers

from mava.environment_loop import ParallelEnvironmentLoop
from mava.utils.loggers import Logger

# from mava.utils.loggers import Logger
from mava.utils.wrapper_utils import RunningStatistics, generate_zeros_from_spec


class EnvironmentLoopStatisticsBase(ParallelEnvironmentLoop):
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
        environment_loop: ParallelEnvironmentLoop,
    ) -> None:
        self._environment = environment_loop._environment
        self._executor = environment_loop._executor
        self._counter = environment_loop._counter
        self._logger = environment_loop._logger
        self._should_update = environment_loop._should_update

    def _compute_step_statistics(self, rewards: Dict[str, float]) -> None:
        raise NotImplementedError

    def _compute_episode_statistics(
        self,
        agent_returns: Dict[str, float],
        episode_return: float,
        episode_length: float,
        steps_per_second: float,
    ) -> None:
        raise NotImplementedError

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

        # Make the first observation.
        self._executor.observe_first(timestep)

        rewards: Dict[str, float] = {}
        episode_returns: Dict[str, float] = {}
        for agent, spec in self._environment.reward_spec().items():
            rewards.update({agent: generate_zeros_from_spec(spec)})
            episode_returns.update({agent: generate_zeros_from_spec(spec)})

        # For evaluation, this keeps track of the total undiscounted reward
        # for each agent accumulated during the episode.
        # multiagent_reward_spec = specs.Array((n_agents,), np.float32)
        # episode_return = tree.map_structure(
        #     generate_zeros_from_spec, multiagent_reward_spec
        # )

        # Run an episode.
        while not timestep.last():

            # Generate an action from the agent's policy and step the environment.
            actions = self._get_actions(timestep)
            timestep = self._environment.step(actions)

            rewards = timestep.reward

            # Have the agent observe the timestep and let the actor update itself.

            self._executor.observe(actions, next_timestep=timestep)

            if self._should_update:
                self._executor.update()

            # Book-keeping.
            episode_steps += 1

            # NOTE (Arnu): fix for when env returns empty dict at end of episode.
            if not rewards:
                rewards = {
                    agent: generate_zeros_from_spec(spec)
                    for agent, spec in self._environment.reward_spec().items()
                }

            self._compute_step_statistics(rewards)

            # Equivalent to: episode_return += timestep.reward
            # We capture the return value because if timestep.reward is a JAX
            # DeviceArray, episode_return will not be mutated in-place. (In all other
            # cases, the returned episode_return will be the same object as the
            # argument episode_return.)
            # episode_return = tree.map_structure(
            #     operator.iadd, episode_return, np.array(list(rewards.values()))
            # )
            episode_returns = {
                agent: episode_returns[agent] + reward
                for agent, reward in rewards.items()
            }

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        episode_return = np.mean(np.array(list(episode_returns.values())))

        self._compute_episode_statistics(
            episode_returns, episode_return, episode_steps, steps_per_second
        )
        self._running_statistics.update({"episode_length": episode_steps})
        self._running_statistics.update(counts)

        return self._running_statistics


class DetailedEpisodeStatistics(EnvironmentLoopStatisticsBase):
    def __init__(self, environment_loop: ParallelEnvironmentLoop):
        super().__init__(environment_loop)

        self._episode_len_stats = RunningStatistics("episode_length")
        self._episode_return_stats = RunningStatistics("episode_return")
        self._steps_per_second_stats = RunningStatistics("steps_per_second")
        self._running_statistics: Dict[str, types.NestedArray] = {
            "mean_episode_length": 0.0,
            "max_episode_length": 0.0,
            "min_episode_length": 0.0,
            "var_episode_length": 0.0,
            "std_episode_length": 0.0,
            "mean_episode_return": 0.0,
            "max_episode_return": 0.0,
            "min_episode_return": 0.0,
            "var_episode_return": 0.0,
            "std_episode_return": 0.0,
            "mean_steps_per_second": 0.0,
            "max_steps_per_second": 0.0,
            "min_steps_per_second": 0.0,
            "var_steps_per_second": 0.0,
            "std_steps_per_second": 0.0,
        }

    def _compute_step_statistics(self, rewards: Dict[str, float]) -> None:
        pass

    def _compute_episode_statistics(
        self,
        agent_returns: Dict[str, float],
        episode_return: float,
        episode_length: float,
        steps_per_second: float,
    ) -> None:

        self._episode_len_stats.push(episode_length)
        self._episode_return_stats.push(episode_return)
        self._steps_per_second_stats.push(steps_per_second)

        self._running_statistics.update(
            {
                "mean_episode_length": self._episode_len_stats.mean(),
                "max_episode_length": self._episode_len_stats.max(),
                "min_episode_length": self._episode_len_stats.min(),
                "var_episode_length": self._episode_len_stats.var(),
                "std_episode_length": self._episode_len_stats.std(),
                "mean_episode_return": self._episode_return_stats.mean(),
                "max_episode_return": self._episode_return_stats.max(),
                "min_episode_return": self._episode_return_stats.min(),
                "var_episode_return": self._episode_return_stats.var(),
                "std_episode_return": self._episode_return_stats.std(),
                "mean_steps_per_second": self._steps_per_second_stats.mean(),
                "max_steps_per_second": self._steps_per_second_stats.max(),
                "min_steps_per_second": self._steps_per_second_stats.min(),
                "var_steps_per_second": self._steps_per_second_stats.var(),
                "std_steps_per_second": self._steps_per_second_stats.std(),
            }
        )


class DetailedPerAgentStatistics(DetailedEpisodeStatistics):
    def __init__(self, environment_loop: ParallelEnvironmentLoop):
        super().__init__(environment_loop)

        # get loop logger data
        loop_label = self._logger._label
        base_dir = self._logger._directory
        (
            to_terminal,
            to_csv,
            to_tensorboard,
            time_delta,
            print_fn,
            time_stamp,
        ) = self._logger._logger_info

        self._agent_running_statistics: Dict[str, Dict[str, float]] = {}
        self._agents_stats: Dict[str, Dict[str, RunningStatistics]] = {
            agent: {} for agent in self._environment.possible_agents
        }
        self._agent_loggers: Dict[str, loggers.Logger] = {}

        # statistics dictionary
        for agent in self._environment.possible_agents:
            agent_label = loop_label + "_" + agent
            self._agent_loggers[agent] = Logger(
                label=agent_label,
                directory=base_dir,
                to_terminal=to_terminal,
                to_csv=to_csv,
                to_tensorboard=to_tensorboard,
                time_delta=time_delta,
                print_fn=print_fn,
                time_stamp=time_stamp,
            )
            self._agents_stats[agent]["return"] = RunningStatistics(
                f"{agent}_episode_return"
            )
            self._agents_stats[agent]["reward"] = RunningStatistics(
                f"{agent}_episode_reward"
            )

    def _compute_step_statistics(self, rewards: Dict[str, float]) -> None:
        for agent, reward in rewards.items():
            self._agents_stats[agent]["reward"].push(reward)
            mean_reward = self._agents_stats[agent]["reward"].mean()
            max_reward = self._agents_stats[agent]["reward"].max()
            min_reward = self._agents_stats[agent]["reward"].min()
            var_reward = self._agents_stats[agent]["reward"].var()
            std_reward = self._agents_stats[agent]["reward"].std()
            self._agent_loggers[agent].write(
                {
                    f"{agent}_mean_episode_reward": mean_reward,
                    f"{agent}_max_episode_reward": max_reward,
                    f"{agent}_min_episode_reward": min_reward,
                    f"{agent}_var_episode_reward": var_reward,
                    f"{agent}_std_episode_reward": std_reward,
                }
            )

    def _compute_episode_statistics(
        self,
        agent_returns: Dict[str, float],
        episode_return: float,
        episode_length: float,
        steps_per_second: float,
    ) -> None:

        self._episode_len_stats.push(episode_length)
        self._episode_return_stats.push(episode_return)
        self._steps_per_second_stats.push(steps_per_second)

        self._running_statistics.update(
            {
                "mean_episode_length": self._episode_len_stats.mean(),
                "max_episode_length": self._episode_len_stats.max(),
                "min_episode_length": self._episode_len_stats.min(),
                "var_episode_length": self._episode_len_stats.var(),
                "std_episode_length": self._episode_len_stats.std(),
                "mean_episode_return": self._episode_return_stats.mean(),
                "max_episode_return": self._episode_return_stats.max(),
                "min_episode_return": self._episode_return_stats.min(),
                "var_episode_return": self._episode_return_stats.var(),
                "std_episode_return": self._episode_return_stats.std(),
                "mean_steps_per_second": self._steps_per_second_stats.mean(),
                "max_steps_per_second": self._steps_per_second_stats.max(),
                "min_steps_per_second": self._steps_per_second_stats.min(),
                "var_steps_per_second": self._steps_per_second_stats.var(),
                "std_steps_per_second": self._steps_per_second_stats.std(),
            }
        )
        for agent, agent_return in agent_returns.items():
            self._agents_stats[agent]["return"].push(agent_return)
            mean_return = self._agents_stats[agent]["return"].mean()
            max_return = self._agents_stats[agent]["return"].max()
            min_return = self._agents_stats[agent]["return"].min()
            var_return = self._agents_stats[agent]["return"].var()
            std_return = self._agents_stats[agent]["return"].std()

            self._agent_loggers[agent].write(
                {
                    f"{agent}_mean_return": mean_return,
                    f"{agent}_max_return": max_return,
                    f"{agent}_min_return": min_return,
                    f"{agent}_var_return": var_return,
                    f"{agent}_std_return": std_return,
                }
            )
