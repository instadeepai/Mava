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
from typing import Dict, List

import numpy as np
from acme.utils import loggers

from mava.environment_loop import ParallelEnvironmentLoop
from mava.utils.loggers import Logger
from mava.utils.wrapper_utils import RunningStatistics


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
        episode_returns: Dict[str, float],
        episode_steps: int,
        start_time: int,
    ) -> None:
        raise NotImplementedError


class DetailedEpisodeStatistics(EnvironmentLoopStatisticsBase):
    def __init__(
        self,
        environment_loop: ParallelEnvironmentLoop,
        summary_stats: List = ["mean", "max", "min", "var", "std", "raw"],
    ):
        super().__init__(environment_loop)
        self._summary_stats = summary_stats
        self._metrics = ["episode_length", "episode_return", "steps_per_second"]
        self._running_statistics: Dict[str, float] = {}
        for metric in self._metrics:
            self.__setattr__(f"_{metric}_stats", RunningStatistics(metric))
            for stat in self._summary_stats:
                self._running_statistics[f"{stat}_{metric}"] = 0.0

    def _compute_step_statistics(self, rewards: Dict[str, float]) -> None:
        pass

    def _compute_episode_statistics(
        self,
        episode_returns: Dict[str, float],
        episode_steps: int,
        start_time: int,
    ) -> None:

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        mean_episode_return = np.mean(np.array(list(episode_returns.values())))

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        self._episode_length_stats.push(episode_steps)
        self._episode_return_stats.push(mean_episode_return)
        self._steps_per_second_stats.push(steps_per_second)

        for metric in self._metrics:
            for stat in self._summary_stats:
                self._running_statistics[f"{stat}_{metric}"] = self.__getattribute__(
                    f"_{metric}_stats"
                ).__getattribute__(stat)()

        self._running_statistics.update({"episode_length": episode_steps})
        self._running_statistics.update(counts)


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
            agent_running_statistics: Dict[str, float] = {}
            self._agents_stats[agent]["reward"].push(reward)
            for stat in self._summary_stats:
                agent_running_statistics[
                    f"{agent}_{stat}_step_reward"
                ] = self._agents_stats[agent]["reward"].__getattribute__(stat)()
            self._agent_loggers[agent].write(agent_running_statistics)

    def _compute_episode_statistics(
        self,
        episode_returns: Dict[str, float],
        episode_steps: int,
        start_time: int,
    ) -> None:

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        mean_episode_return = np.mean(np.array(list(episode_returns.values())))

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        self._episode_length_stats.push(episode_steps)
        self._episode_return_stats.push(mean_episode_return)
        self._steps_per_second_stats.push(steps_per_second)

        for metric in self._metrics:
            for stat in self._summary_stats:
                self._running_statistics[f"{stat}_{metric}"] = self.__getattribute__(
                    f"_{metric}_stats"
                ).__getattribute__(stat)()

        for agent, agent_return in episode_returns.items():
            agent_running_statistics: Dict[str, float] = {}
            self._agents_stats[agent]["return"].push(agent_return)
            for stat in self._summary_stats:
                agent_running_statistics[f"{agent}_{stat}_return"] = self._agents_stats[
                    agent
                ]["return"].__getattribute__(stat)()
            self._agent_loggers[agent].write(agent_running_statistics)

        self._running_statistics.update({"episode_length": episode_steps})
        self._running_statistics.update(counts)
