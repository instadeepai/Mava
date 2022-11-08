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


import copy

import logging
import time
from typing import Any, Dict, Tuple

import acme
import dm_env
import jax
import numpy as np
import tensorflow as tf
from acme.utils import counting, loggers

import mava
from mava.utils.checkpointing_utils import update_best_checkpoint, update_to_best_net
from mava.utils.training_utils import check_count_condition
from mava.utils.wrapper_utils import generate_zeros_from_spec


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
        """Parallel environment loop init

        Args:
            environment: an environment
            executor: a Mava executor
            counter: an optional counter. Defaults to None.
            logger: an optional counter. Defaults to None.
            should_update: should update. Defaults to True.
            label: optional label. Defaults to "parallel_environment_loop".
        """
        # Internalize agent and environment.
        self._environment = environment
        self._executor = executor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)

        self._should_update = should_update
        self._running_statistics: Dict[str, float] = {}

        # We need this to schedule evaluation/test runs
        self._last_evaluator_run_t = -1

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

    def record_counts(self, episode_steps: int) -> counting.Counter:
        """Record latest counts"""
        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)
        return counts

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
            counts = self.record_counts(episode_steps)

            # Collect the results and combine with counts.
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                "episode_length": episode_steps,
                "mean_episode_return": np.mean(list(episode_returns.values())),
                "steps_per_second": steps_per_second,
            }
            result.update(counts)
            return result

    def run_episode_and_log(self) -> loggers.LoggingData:
        """Run an episode and log the results"""

        results = self.run_episode()
        self._logger.write(results)
        return results

    def run(self) -> None:
        """Run the environment loop."""

        def should_run_loop(eval_interval_condition: Tuple) -> bool:
            """Check if the eval loop should run in current step.

            Args:
                eval_interval_condition : tuple containing interval key and count.

            Returns:
                a bool indicating if eval should run.
            """
            should_run_loop = False
            eval_interval_key, eval_interval_count = eval_interval_condition
            counts = self._executor.store.executor_counts

            if counts:
                count = counts[eval_interval_key]
                # We run eval loops around every eval_interval_count (not exactly
                # every eval_interval_count due to latency in getting updated counts).
                should_run_loop = (
                    (count - self._last_evaluator_run_t) / eval_interval_count
                ) >= 1.0
                if should_run_loop:
                    self._last_evaluator_run_t = int(count)
                    print(
                        "Running eval loop at executor step: "
                        + f"{self._last_evaluator_run_t}"
                    )

            return should_run_loop

        environment_loop_schedule = self._executor._evaluator and (
            self._executor.store.evaluation_interval is not None
        )

        if environment_loop_schedule:
            eval_interval_condition = check_count_condition(
                self._executor.store.evaluation_interval
            )
            eval_duration_condition = check_count_condition(
                self._executor.store.evaluation_duration
            )
            evaluation_duration = eval_duration_condition[1]

        def run_evaluation(results: Any) -> None:
            """Calculate the absolute metric"""
            tf.print("Calculating the absolute metric")
            for metric in self._executor.store.metrics_checkpoint.keys():
                used_results = copy.deepcopy(results)
                # update the evaluator network
                for agent_net_key in self._executor.store.networks.keys():
                    self._executor.store.networks[
                        agent_net_key
                    ].policy_params = copy.deepcopy(
                        self._executor.store.best_checkpoint[metric][
                            f"policy_network-{agent_net_key}"
                        ]
                    )
                    self._executor.store.networks[
                        agent_net_key
                    ].critic_params = copy.deepcopy(
                        self._executor.store.best_checkpoint[metric][
                            f"critic_network-{agent_net_key}"
                        ]
                    )
                    self._executor.store.policy_opt_states[
                        agent_net_key
                    ] = copy.deepcopy(
                        self._executor.store.best_checkpoint[metric][
                            f"policy_opt_state-{agent_net_key}"
                        ]
                    )
                    self._executor.store.critic_opt_states[
                        agent_net_key
                    ] = copy.deepcopy(
                        self._executor.store.best_checkpoint[metric][
                            f"critic_opt_state-{agent_net_key}"
                        ]
                    )
                for _ in range(self._executor.store.absolute_metric_duration):
                    # Add consecutive evaluation run data
                    result = self.run_episode()
                    # Sum results for computing mean after all evaluation runs.
                used_results = jax.tree_map(lambda x, y: x + y, used_results, result)
                # compute the mean over all evaluation runs
                used_results = jax.tree_map(
                    lambda x: x / evaluation_duration, used_results
                )
                # Check for extra logs
                if hasattr(self._environment, "get_interval_stats"):
                    used_results.update(self._environment.get_interval_stats())
                tf.print(
                    f"Absolute metric results for {metric} is equal to {used_results[metric]}"
                )
                tf.print(f"Additional results {used_results}")
            tf.print("Terminate the system")
            self._executor.store.executor_parameter_client.set_and_wait(
                {"terminate": True}
            )

        def step_executor() -> None:
            if (not environment_loop_schedule) or (
                should_run_loop(eval_interval_condition)
            ):
                if environment_loop_schedule:
                    results = self.run_episode()
                    if (
                        results["executor_steps"] >= 20000
                        and self._executor.store.absolute_metric
                    ):
                        tf.print(
                            f"Executor has reached {results['executor_steps']} steps"
                        )
                        run_evaluation(results=results)
                    else:
                        # Get first result dictionary
                        for _ in range(evaluation_duration - 1):
                            # Add consecutive evaluation run data
                            result = self.run_episode()
                            # Sum results for computing mean after all evaluation runs.
                            results = jax.tree_map(lambda x, y: x + y, results, result)
                        # compute the mean over all evaluation runs
                        results = jax.tree_map(
                            lambda x: x / evaluation_duration, results
                        )
                        # Check for extra logs
                        if hasattr(self._environment, "get_interval_stats"):
                            results.update(self._environment.get_interval_stats())
                        self._logger.write(results)
                        if self._executor.store.checkpoint_best_perf:
                            # Best_performance_update
                            for (
                                metric,
                                best_performance,
                            ) in self._executor.store.metrics_checkpoint.items():
                                assert (
                                    metric in results.keys()
                                ), f"The metric chosen to checkpoint it best performance doesn't exist.\
                                    This experiment has only the following metrics {results.keys()}"

                                if (
                                    best_performance is None
                                    or best_performance < results[metric]  # type: ignore
                                ):
                                    self._executor.store.metrics_checkpoint[
                                        metric
                                    ] = update_best_checkpoint(
                                        self._executor, results, metric
                                    )
                else:
                    result = self.run_episode()
                    # Log the given results.
                    self._logger.write(result)
            else:
                # Note: We assume that the evaluator will be running less
                # than once per second.
                time.sleep(1)
            # We need to get the latest counts if we are using eval intervals.
            if environment_loop_schedule:
                self._executor.force_update()

        while True:
            try:
                step_executor()

            except Exception as e:
                if self._executor._evaluator:
                    logging.exception(
                        f"{e}: Experiment terminated due to an error on the evaluator."
                    )
                    self._executor.store.executor_parameter_client.set_and_wait(
                        {"terminate": True}
                    )
                else:
                    logging.exception(f"{e}: an executor failed.")
                    self._executor.store.executor_parameter_client.add_and_wait(
                        {"num_executor_failed": 1}
                    )
                self._executor.force_update()
                break
