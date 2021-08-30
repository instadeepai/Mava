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
from typing import Any, Dict, List, Tuple, Union

import dm_env
import matplotlib.pyplot as plt
import numpy as np
from acme.utils import counting, loggers, paths
from acme.wrappers.video import make_animation

try:
    from array2gif import write_gif
except ModuleNotFoundError:
    pass

import mava
from mava.environment_loop import ParallelEnvironmentLoop, SequentialEnvironmentLoop
from mava.utils.loggers import Logger
from mava.utils.wrapper_utils import RunningStatistics


class EnvironmentLoopStatisticsBase:
    """
    A base stats class that acts as a MARL environment loop wrapper.
    """

    def __init__(
        self,
        environment_loop: Union[ParallelEnvironmentLoop, SequentialEnvironmentLoop],
    ) -> None:
        self._environment_loop = environment_loop
        self._override_environment_loop_stats_methods()
        self._running_statistics: Dict[str, float] = {}

    def _compute_step_statistics(self, rewards: Dict[str, float]) -> None:
        raise NotImplementedError

    def _compute_episode_statistics(
        self,
        episode_returns: Dict[str, float],
        episode_steps: int,
        start_time: float,
    ) -> None:
        raise NotImplementedError

    def _get_running_stats(self) -> Dict:
        return self._running_statistics

    def _override_environment_loop_stats_methods(self) -> None:
        self._environment_loop._compute_episode_statistics = (  # type: ignore
            self._compute_episode_statistics
        )
        self._environment_loop._compute_step_statistics = (  # type: ignore
            self._compute_step_statistics
        )
        self._environment_loop._get_running_stats = (  # type: ignore
            self._get_running_stats
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment_loop, name)


class DetailedEpisodeStatistics(EnvironmentLoopStatisticsBase):
    """
    A stats class that acts as a MARL environment loop wrapper
    and overwrites _compute_episode_statistics.
    """

    def __init__(
        self,
        environment_loop: Union[ParallelEnvironmentLoop, SequentialEnvironmentLoop],
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
        start_time: float,
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
    """
    A stats class that acts as a MARL environment loop wrapper
    and overwrites _compute_episode_statistics and _compute_step_statistics.
    """

    def __init__(
        self,
        environment_loop: Union[ParallelEnvironmentLoop, SequentialEnvironmentLoop],
    ):
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
        start_time: float,
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


class MonitorParallelEnvironmentLoop(ParallelEnvironmentLoop):
    """A MARL environment loop.
    This records a gif/video every `record_every` eval episodes.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        executor: mava.core.Executor,
        filename: str = "agents",
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        should_update: bool = True,
        label: str = "parallel_environment_loop",
        record_every: int = 1000,
        path: str = "~/mava",
        fps: int = 15,
        counter_str: str = "evaluator_episodes",
        format: str = "video",
        figsize: Union[float, Tuple[int, int]] = (360, 640),
    ):
        assert (
            format == "gif" or format == "video"
        ), "Only gif and video format are supported."

        # Internalize agent and environment.
        super().__init__(
            environment=environment,
            executor=executor,
            counter=counter,
            logger=logger,
            should_update=should_update,
            label=label,
        )
        self._record_every = record_every
        self._path = paths.process_path(path, "recordings", add_uid=False)
        self._filename = filename
        self._record_current_episode = False
        self._fps = fps
        self._frames: List = []

        self._parent_environment_step = self._environment.step
        self._environment.step = self.step

        self._parent_environment_reset = self._environment.reset
        self._environment.reset = self.reset

        self._counter_str = counter_str

        self._format = format
        self._figsize = figsize

    def step(self, action: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        timestep = self._parent_environment_step(action)
        self._append_frame()
        return timestep

    def _retrieve_render(self) -> np.ndarray:
        render = None
        try:
            if self._format == "video":
                render = self._environment.render(mode="rgb_array")
            elif self._format == "gif":
                render = np.transpose(
                    self._environment.render(mode="rgb_array"), axes=(1, 0, 2)
                )
        except Exception as ex:
            print(f"Render frames exception: {ex}")
            pass
        return render

    def _append_frame(self) -> None:
        """Appends a frame to the sequence of frames."""
        counts = self._counter.get_counts()
        counter = counts.get(self._counter_str)
        if counter and (counter % self._record_every == 0):
            self._frames.append(self._retrieve_render())

    def reset(self) -> dm_env.TimeStep:
        if self._frames:
            self._write_frames()
        timestep = self._parent_environment_reset()
        return timestep

    def _write_frames(self) -> None:
        counts = self._counter.get_counts()
        counter = counts.get(self._counter_str)
        path = f"{self._path}/{self._filename}_{counter}_eval_episode"
        try:
            if self._format == "video":
                self._save_video(path)
            elif self._format == "gif":
                self._save_gif(path)
        except Exception as ex:
            print(f"Write frames exception: {ex}")
            pass
        self._frames = []
        # Clear matplotlib figures in memory
        plt.close("all")

    def _save_video(self, path: str) -> None:
        video = make_animation(self._frames, self._fps, self._figsize).to_html5_video()
        with open(f"{path}.html", "w") as f:
            f.write(video)

    def _save_gif(self, path: str) -> None:
        write_gif(
            self._frames,
            f"{path}.gif",
            fps=self._fps,
        )
