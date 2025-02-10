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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from colorama import Fore, Style
from jax import tree
from jax.sharding import Sharding
from jumanji.types import TimeStep

# todo: remove the ppo dependencies when we make sebulba for other systems
from mava.systems.ppo.types import HiddenStates, Params
from mava.types import MavaTransition, Metrics

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
