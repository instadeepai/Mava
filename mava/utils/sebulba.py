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
import time
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional

import jax
import jax.numpy as jnp
from colorama import Fore, Style
from jax import tree
from jax.sharding import Sharding
from jumanji.types import TimeStep

# todo: remove the ppo dependencies when we make sebulba for other systems
from mava.systems.ppo.types import Params, PPOTransition
from mava.types import Metrics

QUEUE_PUT_TIMEOUT = 100


class ThreadLifetime:
    """Simple class for a mutable boolean that can be used to signal a thread to stop."""

    def __init__(self) -> None:
        self._stop = False

    def should_stop(self) -> bool:
        return self._stop

    def stop(self) -> None:
        self._stop = True


@jax.jit
def _stack_trajectory(trajectory: List[PPOTransition]) -> PPOTransition:
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

    def __init__(self, max_size: int, learner_sharding: Sharding, lifetime: ThreadLifetime):
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
        self.lifetime = lifetime

    def run(self) -> None:
        """This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self.lifetime.should_stop():
            try:
                start_condition, end_condition = self.tickets_queue.get(timeout=1)
                with end_condition:
                    with start_condition:
                        start_condition.notify()
                    end_condition.wait()
            except queue.Empty:
                continue

    def put(
        self, traj: Sequence[PPOTransition], timestep: TimeStep, metrics: Tuple[Dict, List[Dict]]
    ) -> None:
        """Put a trajectory on the queue to be consumed by the learner."""
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()  # wait to be allowed to start

        # [Transition(num_envs)] * rollout_len -> Transition[done=(num_envs, rollout_len, ...)]
        traj = _stack_trajectory(traj)
        traj, timestep = jax.device_put((traj, timestep), device=self.sharding)

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
                (traj, timestep, time_dict, episode_metrics),
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
    ) -> Tuple[PPOTransition, TimeStep, Dict, Metrics]:
        """Get a trajectory from the pipeline."""
        return self._queue.get(block, timeout)  # type: ignore

    def clear(self) -> None:
        """Clear the pipeline."""
        while not self._queue.empty():
            try:
                self._queue.get(block=False)
            except queue.Empty:
                break


class ParamsSource(threading.Thread):
    """A `ParamSource` is a component that allows networks params to be passed from a
    `Learner` component to `Actor` components.
    """

    def __init__(self, init_value: Params, device: jax.Device, lifetime: ThreadLifetime):
        super().__init__(name=f"ParamsSource-{device.id}")
        self.value: Params = jax.device_put(init_value, device)
        self.device = device
        self.new_value: queue.Queue = queue.Queue()
        self.lifetime = lifetime

    def run(self) -> None:
        """This function is responsible for updating the value of the `ParamSource` when a new value
        is available.
        """
        while not self.lifetime.should_stop():
            try:
                waiting = self.new_value.get(block=True, timeout=1)
                self.value = jax.device_put(waiting, self.device)
            except queue.Empty:
                continue

    def update(self, new_params: Params) -> None:
        """Update the value of the `ParamSource` with a new value.

        Args:
            new_params: The new value to update the `ParamSource` with.
        """
        self.new_value.put(new_params)

    def get(self) -> Params:
        """Get the current value of the `ParamSource`."""
        return self.value


class RecordTimeTo:
    """Context manager to record the runtime in a `with` block"""

    def __init__(self, to: Any):
        self.to = to

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, *args: Any) -> None:
        end = time.monotonic()
        self.to.append(end - self.start)



# Modified from https://github.com/instadeepai/sebulba/blob/main/sebulba/core.py
class OfflinePipeline(threading.Thread): #todo why dosen't sotix keep the latest metrics?
    """
    The `Pipeline` shards trajectories into learner devices,
    ensuring trajectories are consumed in the right order to avoid being off-policy
    and limit the max number of samples in device memory at one time to avoid OOM issues.
    """

    def __init__(self, max_size: int, learner_sharding: Sharding, lifetime: ThreadLifetime,buffer_add, buffer_sample, buffer_state, key, rate_limiter):
        """
        Initializes the pipeline with a maximum size and the devices to shard trajectories across.

        Args:
            max_size: The maximum number of trajectories to keep in the pipeline.
            learner_sharding: The sharding used for the learner's update function.
            lifetime: A `ThreadLifetime` which is used to stop this thread.
        """
        super().__init__(name="Pipeline")
        self.cpu = jax.devices("cpu")[0]
        
        self.sharding = learner_sharding
        self.tickets_queue: queue.Queue = queue.Queue()
        self._queue: queue.Queue = queue.Queue()
        self.lifetime = lifetime
        self.last_actor_metrics = None
        
        #buffer util
        self.move_to_device = lambda tree: jax.tree.map(lambda x: jax.device_put(x, self.cpu), tree)
        self.buffer_state =  buffer_state
        self.buffer_add = jax.jit(buffer_add, device=self.cpu)
        self.buffer_sample = jax.jit(buffer_sample, device=self.cpu)
        self.key  = key

        #rate limiter
        self.rate_limiter = rate_limiter

        
    def run(self) -> None:
        """This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self.lifetime.should_stop():
            try:
                start_condition, end_condition = self.tickets_queue.get(timeout=1)
                with end_condition:
                    with start_condition:
                        start_condition.notify()
                    end_condition.wait()
            except queue.Empty:
                continue

    def put(self, traj: Sequence[PPOTransition], metrics: Tuple) -> None:
        """Put a trajectory on the queue to be consumed by the learner."""
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()  # wait to be allowed to start      
        
        try:    #todo look at this blocking from old and stoix
            self.rate_limiter.await_can_insert(timeout=60) 
        except TimeoutError:
            print(
                f"{Fore.RED}{Style.BRIGHT}Actor has timed out on insertion, "
                f"this should not happen. A deadlock might be occurring{Style.RESET_ALL}"
            )
        if self.buffer_state.is_full:
                self.rate_limiter.delete()

        # [Transition(num_envs)] * rollout_len -> Transition[done=(num_envs, rollout_len, ...)]
        traj = _stack_trajectory(traj)
        traj = jax.device_put(traj, device=self.sharding)

        time_dict, episode_metrics = metrics
        # [{'metric1' : value1, ...} * rollout_len -> {'metric1' : [value1, value2, ...], ...}
        episode_metrics = _stack_trajectory(episode_metrics)


        self.buffer_state = self.buffer_add(self.buffer_state, traj)
        self.rate_limiter.insert()
        self._queue.put((time_dict, episode_metrics))
    
        with end_condition:
            end_condition.notify()  # notify that we have finished

        

    def qsize(self) -> int:
        """Returns the number of trajectories in the pipeline."""
        return self._queue.qsize()

    def get(
        self, block: bool = True, timeout: Union[float, None] = None
    ) -> Tuple[PPOTransition, TimeStep, Dict]:
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

        # sample the data
        sampled_batch = self.buffer_sample(self.buffer_state, sample_key).experience
        self.rate_limiter.sample()
        sampled_batch = jax.device_put(sampled_batch, device=self.sharding)
        if not self._queue.empty():
            self.last_actor_metrics = self._queue.get()

        return sampled_batch, self.last_actor_metrics

    def clear(self) -> None:
        """Clear the pipeline."""
        while not self._queue.empty():
            try:
                self._queue.get(block=False)
            except queue.Empty:
                break



# from https://github.com/EdanToledo/Stoix/blob/feat/sebulba-dqn/stoix/utils/rate_limiters.py
class RateLimiter:
    def __init__(
        self, samples_per_insert: float, min_size_to_sample: int, min_diff: float, max_diff: float
    ):
        assert min_size_to_sample > 0, "min_size_to_sample must be greater than 0"
        assert samples_per_insert > 0, "samples_per_insert must be greater than 0"

        self.samples_per_insert = samples_per_insert
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.min_size_to_sample = min_size_to_sample

        self.inserts = 0
        self.samples = 0
        self.deletes = 0

        self.mutex = threading.Lock()
        self.condition = threading.Condition(self.mutex)

    def num_inserts(self) -> int:
        """Returns the number of inserts."""
        with self.mutex:
            return self.inserts

    def num_samples(self) -> int:
        """Returns the number of samples."""
        with self.mutex:
            return self.samples

    def num_deletes(self) -> int:
        """Returns the number of deletes."""
        with self.mutex:
            return self.deletes

    def insert(self) -> None:
        """Increment the number of inserts and notify all waiting threads."""
        with self.mutex:
            self.inserts += 1
            self.condition.notify_all()  # Notify all waiting threads

    def delete(self) -> None:
        """Increment the number of deletes and notify all waiting threads."""
        with self.mutex:
            self.deletes += 1
            self.condition.notify_all()  # Notify all waiting threads

    def sample(self) -> None:
        """Increment the number of samples and notify all waiting threads."""
        with self.mutex:
            self.samples += 1
            self.condition.notify_all()  # Notify all waiting threads

    def can_insert(self, num_inserts: int) -> bool:
        """Check if the caller can insert `num_inserts` items."""
        # Assume lock is already held by the caller
        if num_inserts <= 0:
            return False
        if self.inserts + num_inserts - self.deletes <= self.min_size_to_sample:
            return True
        diff = (num_inserts + self.inserts) * self.samples_per_insert - self.samples
        return diff <= self.max_diff

    def can_sample(self, num_samples: int) -> bool:
        """Check if the caller can sample `num_samples` items."""
        # Assume lock is already held by the caller
        if num_samples <= 0:
            return False
        if self.inserts - self.deletes < self.min_size_to_sample:
            return False
        diff = self.inserts * self.samples_per_insert - self.samples - num_samples
        return diff >= self.min_diff

    def await_can_insert(self, num_inserts: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until the caller can insert `num_inserts` items."""
        with self.condition:
            result = self.condition.wait_for(lambda: self.can_insert(num_inserts), timeout)
            if not result:
                raise TimeoutError(f"Timeout occurred while waiting to insert {num_inserts} items.")
            return result

    def await_can_sample(self, num_samples: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until the caller can sample `num_samples` items."""
        with self.condition:
            result = self.condition.wait_for(lambda: self.can_sample(num_samples), timeout)
            if not result:
                raise TimeoutError(f"Timeout occurred while waiting to sample {num_samples} items.")
            return result

    def __repr__(self) -> str:
        return (
            f"RateLimiter(samples_per_insert={self.samples_per_insert}, "
            f"min_size_to_sample={self.min_size_to_sample}, "
            f"min_diff={self.min_diff}, max_diff={self.max_diff})"
        )


class MinSize(RateLimiter):
    """Block sample calls unless replay contains `min_size_to_sample`.

    This limiter blocks all sample calls when the replay contains less than
    `min_size_to_sample` items, and accepts all sample calls otherwise.
    """

    def __init__(self, min_size_to_sample: int):
        if min_size_to_sample < 1:
            raise ValueError(
                f"min_size_to_sample ({min_size_to_sample}) must be a positive integer"
            )

        super().__init__(
            samples_per_insert=1.0,
            min_size_to_sample=min_size_to_sample,
            min_diff=-sys.float_info.max,
            max_diff=sys.float_info.max,
        )


class SampleToInsertRatio(RateLimiter):
    """Maintains a specified ratio between samples and inserts.

    The limiter works in two stages:

      Stage 1. Size of table is lt `min_size_to_sample`.
      Stage 2. Size of table is ge `min_size_to_sample`.

    During stage 1 the limiter works exactly like MinSize, i.e. it allows
    all insert calls and blocks all sample calls. Note that it is possible to
    transition into stage 1 from stage 2 when items are removed from the table.

    During stage 2 the limiter attempts to maintain the `samples_per_insert`
    ratio between the samples and inserts. This is done by
    measuring the `error`, calculated as:

      error = number_of_inserts * samples_per_insert - number_of_samples

    and making sure that `error` stays within `allowed_range`. Any operation
    which would move `error` outside of the `allowed_range` is blocked.
    Such approach allows for small deviation from a target `samples_per_insert`,
    which eliminates excessive blocking of insert/sample operations and improves
    performance.

    If `error_buffer` is a tuple of two numbers then `allowed_range` is defined as

      (error_buffer[0], error_buffer[1])

    When `error_buffer` is a single number then the range is defined as

      (
        min_size_to_sample * samples_per_insert - error_buffer,
        min_size_to_sample * samples_per_insert + error_buffer
      )
    """

    def __init__(
        self,
        samples_per_insert: float,
        min_size_to_sample: int,
        error_buffer: Union[float, Tuple[float, float]],
    ):
        """Constructor of SampleToInsertRatio.

        Args:
          samples_per_insert: The average number of times the learner should sample
            each item in the replay buffer during the item's entire lifetime.
          min_size_to_sample: The minimum number of items that the table must
            contain  before transitioning into stage 2.
          error_buffer: Maximum size of the "error" before calls should be blocked.
            When a single value is provided then inferred range is
              (
                min_size_to_sample * samples_per_insert - error_buffer,
                min_size_to_sample * samples_per_insert + error_buffer
              )
            The offset is added so that the error tracked is for the insert/sample
            ratio only takes into account operations occurring AFTER stage 1. If a
            range (two float tuple) then the values are used without any offset.

        Raises:
          ValueError: If error_buffer is smaller than max(1.0, samples_per_inserts).
        """
        if isinstance(error_buffer, float) or isinstance(error_buffer, int):
            offset = samples_per_insert * min_size_to_sample
            min_diff = offset - error_buffer
            max_diff = offset + error_buffer
        else:
            min_diff, max_diff = error_buffer

        if samples_per_insert <= 0:
            raise ValueError(f"samples_per_insert ({samples_per_insert}) must be > 0")

        if max_diff - min_diff < 2 * max(1.0, samples_per_insert):
            raise ValueError(
                "The size of error_buffer must be >= max(1.0, samples_per_insert) as "
                "smaller values could completely block samples and/or insert calls."
            )

        if max_diff < samples_per_insert * min_size_to_sample:
            print(
                f"{Fore.YELLOW}{Style.BRIGHT}The range covered by error_buffer is below "
                "samples_per_insert * min_size_to_sample. If the sampler cannot "
                "sample concurrently, this will result in a deadlock as soon as "
                f"min_size_to_sample items have been inserted.{Style.RESET_ALL}"
            )
        if min_diff > samples_per_insert * min_size_to_sample:
            raise ValueError(
                "The range covered by error_buffer is above "
                "samples_per_insert * min_size_to_sample. This will result in a "
                "deadlock as soon as min_size_to_sample items have been inserted."
            )

        if min_size_to_sample < 1:
            raise ValueError(
                f"min_size_to_sample ({min_size_to_sample}) must be a positive integer"
            )

        super().__init__(
            samples_per_insert=samples_per_insert,
            min_size_to_sample=min_size_to_sample,
            min_diff=min_diff,
            max_diff=max_diff,
        )