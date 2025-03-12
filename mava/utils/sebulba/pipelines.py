# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

import queue
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from flashbax import make_trajectory_buffer
from jax import tree
from jax.sharding import Sharding
from jumanji.types import TimeStep
from omegaconf import DictConfig

# todo: remove the ppo dependencies when we make sebulba for other systems
from mava.systems.ppo.types import HiddenStates
from mava.types import MavaTransition, Metrics
from mava.utils.sebulba.rate_limiters import RateLimiter

QUEUE_PUT_TIMEOUT = 100


@jax.jit
def _stack_trajectory(trajectory: List[MavaTransition]) -> MavaTransition:
    """Stack a list of parallel_env transitions into a single
    transition of shape [rollout_len, num_envs, ...]."""
    return tree.map(lambda *x: jnp.stack(x, axis=0).swapaxes(0, 1), *trajectory)  # type: ignore


# Modified from https://github.com/instadeepai/sebulba/blob/main/sebulba/core.py
class Pipeline(threading.Thread):
    """
    The `Pipeline` shards trajectories into learner devices,
    ensuring trajectories are consumed in the right order to avoid being off-policy
    and limit the max number of samples in device memory at one time to avoid OOM issues.
    """

    def __init__(self, max_size: int, learner_sharding: Sharding):
        """
        Initializes the pipeline with a maximum size and the devices to shard trajectories across.

        Args:
            max_size: The maximum number of trajectories to keep in the pipeline.
            learner_sharding: The sharding used for the learner's update function.
            lifetime: A `ThreadLifetime` which is used to stop this thread.
        """
        super().__init__(name="Pipeline")

        self.sharding = learner_sharding
        self.tickets_queue: queue.Queue = queue.Queue()
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._should_stop = False

    def run(self) -> None:
        """This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self._should_stop:
            try:
                start_condition, end_condition = self.tickets_queue.get(timeout=1)
                with end_condition:
                    with start_condition:
                        start_condition.notify()
                    end_condition.wait()
            except queue.Empty:
                continue

    def put(
        self,
        traj: Sequence[MavaTransition],
        metrics: Tuple[Dict, List[Dict]],
        final_timestep: Tuple[TimeStep, Optional[HiddenStates]],
    ) -> None:
        """Put a trajectory on the queue to be consumed by the learner."""
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()  # wait to be allowed to start

        # [Transition(num_envs)] * rollout_len -> Transition[done=(num_envs, rollout_len, ...)]
        traj = _stack_trajectory(traj)
        traj, final_timestep = jax.device_put((traj, final_timestep), device=self.sharding)

        time_dict, episode_metrics = metrics
        # [{'metric1' : value1, ...} * rollout_len -> {'metric1' : [value1, value2, ...], ...}
        episode_metrics = _stack_trajectory(episode_metrics)

        # We block on the `put` to ensure that actors wait for the learners to catch up.
        # This ensures two things:
        #  The actors don't get too far ahead of the learners, which could lead to off-policy data.
        #  The actors don't "waste" samples by generating samples that the learners can't consume.
        # However, we put a timeout of 100 seconds to avoid deadlocks in case the learner
        # is not consuming the data. This is a safety measure and should not normally occur.
        # We use a try-finally so the lock is released even if an exception is raised.
        try:
            self._queue.put(
                (traj, time_dict, episode_metrics, final_timestep),
                block=True,
                timeout=QUEUE_PUT_TIMEOUT,
            )
        except queue.Full:
            print(
                f"{Fore.RED}{Style.BRIGHT}Pipeline is full and actor has timed out, "
                f"this should not happen. A deadlock might be occurring{Style.RESET_ALL}"
            )
        finally:
            with end_condition:
                end_condition.notify()  # notify that we have finished

    def qsize(self) -> int:
        """Returns the number of trajectories in the pipeline."""
        return self._queue.qsize()

    def get(
        self, block: bool = True, timeout: Union[float, None] = None
    ) -> Tuple[MavaTransition, Dict, Metrics, Tuple[TimeStep, Optional[HiddenStates]]]:
        """Get a trajectory from the pipeline."""
        return self._queue.get(block, timeout)  # type: ignore

    def clear(self) -> None:
        """Clear the pipeline."""
        while not self._queue.empty():
            try:
                self._queue.get(block=False)
            except queue.Empty:
                break

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._should_stop = True


class OffPolicyPipeline(threading.Thread):
    """
    The `Pipeline` shards trajectories into learner devices,
    ensuring trajectories are consumed in the right order to avoid being off-policy
    and limit the max number of samples in device memory at one time to avoid OOM issues.
    """

    def __init__(
        self,
        config: DictConfig,
        learner_sharding: Sharding,
        key: jax.random.PRNGKey,
        rate_limiter: RateLimiter,
        init_transition: MavaTransition,
    ):
        """
        Initializes the pipeline with a maximum size and the devices to shard trajectories across.

        Args:
            config: Configuration settings for buffers.
            learner_sharding: The sharding used for the learner's update function.
            key: The PRNG key for stochasticity.
            rate_limiter: A `RateLimiter` Used to manage how often we are allowed to
            sample from the buffers.
            init_transition : A sample trasition used to initialize the buffers.
            lifetime: A `ThreadLifetime` which is used to stop this thread.
        """
        super().__init__(name="Pipeline")
        self.cpu = jax.devices("cpu")[0]

        self.tickets_queue: queue.Queue = queue.Queue()
        # Only keep the latest 100 metrics, otherwise too many metrics are added we risk an OOM
        self.metrics_queue: queue.Queue = queue.Queue(maxsize=100)
        self._should_stop = False

        self.num_buffers = len(config.arch.actor_device_ids) * config.arch.n_threads_per_executor
        self.rate_limiter = rate_limiter
        self.sharding = learner_sharding
        self.key = key

        assert config.system.sample_batch_size % self.num_buffers == 0, (
            f"The sample batch size ({config.system.sample_batch_size}) must be divisible "
            f"by the total number of actors ({self.num_buffers})."
        )

        # Setup Buffers
        rb = make_trajectory_buffer(
            sample_sequence_length=config.system.sample_sequence_length + 1,
            period=1,
            add_batch_size=config.arch.num_envs,
            sample_batch_size=config.system.sample_batch_size // self.num_buffers,
            max_length_time_axis=config.system.buffer_size,
            min_length_time_axis=config.system.min_buffer_size,
        )
        self.buffer_states = [rb.init(init_transition) for _ in range(self.num_buffers)]

        # Setup functions
        self.buffer_add = jax.jit(rb.add, device=self.cpu)
        self.buffer_sample = jax.jit(rb.sample, device=self.cpu)

    def run(self) -> None:
        """This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self._should_stop:
            try:
                start_condition, end_condition = self.tickets_queue.get(timeout=1)
                with end_condition:
                    with start_condition:
                        start_condition.notify()
                    end_condition.wait()
            except queue.Empty:
                continue

    def put(self, traj: Sequence[MavaTransition], metrics: Tuple, actor_id: int) -> None:
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()

        try:
            self.rate_limiter.await_can_insert(timeout=QUEUE_PUT_TIMEOUT)
        except TimeoutError:
            print(
                f"{Fore.RED}{Style.BRIGHT}Actor has timed out on insertion, "
                f"this should not happen. A deadlock might be occurring{Style.RESET_ALL}"
            )

        traj = jax.device_get(traj)
        # [Transition(num_envs)] * rollout_len -> Transition[done=(num_envs, rollout_len, ...)]
        traj = _stack_trajectory(traj)

        time_dict, episode_metrics = metrics
        # [{'metric1' : value1, ...] * rollout_len -> {'metric1' : [value1, value2, ...], ...}
        episode_metrics = _stack_trajectory(episode_metrics)

        self.buffer_states[actor_id] = self.buffer_add(self.buffer_states[actor_id], traj)

        if self.metrics_queue.full():
            self.metrics_queue.get()  # remove the oldest entry

        self.metrics_queue.put((time_dict, episode_metrics))

        self.rate_limiter.insert(1 / self.num_buffers)

        with end_condition:
            end_condition.notify()  # notify that we have finished

    def get(self, timeout: Union[float, None] = None) -> Tuple[MavaTransition, Any]:
        """Get a trajectory from the pipeline."""
        self.key, sample_key = jax.random.split(self.key)

        # wait until we can sample the data
        try:
            self.rate_limiter.await_can_sample(timeout=timeout)
        except TimeoutError:
            print(
                f"{Fore.RED}{Style.BRIGHT}Learner has timed out on sampling, "
                f"this should not happen. A deadlock might be occurring{Style.RESET_ALL}"
            )

        # Sample the data
        # Potential deadlock risk here. Although it hasn't occurred during testing.
        # if an unexplained deadlock happens, it is likely due to this section.
        sampled_batch: List[MavaTransition] = [
            self.buffer_sample(state, sample_key).experience for state in self.buffer_states
        ]
        transitions: MavaTransition = tree.map(lambda *x: np.concatenate(x), *sampled_batch)
        transitions = jax.device_put(transitions, device=self.sharding)

        self.rate_limiter.sample()

        if not self.metrics_queue.empty():
            return transitions, self.metrics_queue.get()

        return transitions, (None, None)

    def clear(self) -> None:
        """Clear the pipeline."""
        while not self.metrics_queue.empty():
            try:
                self.metrics_queue.get(block=False)
            except queue.Empty:
                break

    def qsize(self) -> int:
        """Returns the number of trajectories in the pipeline."""
        return self.metrics_queue.qsize()

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._should_stop = True
