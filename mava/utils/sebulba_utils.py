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
from typing import Any, Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from chex import Array

from mava.systems.ppo.types import Params, PPOTransition  # todo: remove the ppo dependencies
from mava.types import Observation, ObservationGlobalState


# Copied from https://github.com/instadeepai/sebulba/blob/main/sebulba/core.py
class Pipeline(threading.Thread):
    """
    The `Pipeline` shards trajectories into `learner_devices`,
    ensuring trajectories are consumed in the right order to avoid being off-policy
    and limit the max number of samples in device memory at one time to avoid OOM issues.
    """

    def __init__(self, max_size: int, learner_devices: List[jax.Device]):
        """
        Initializes the pipeline with a maximum size and the devices to shard trajectories across.

        Args:
            max_size: The maximum number of trajectories to keep in the pipeline.
            learner_devices: The devices to shard trajectories across.
        """
        super().__init__(name="Pipeline")
        self.learner_devices = learner_devices
        self.tickets_queue: queue.Queue = queue.Queue()
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)

    def run(self) -> None:
        """
        This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while True:  # todo Thread lifetime
            start_condition, end_condition = self.tickets_queue.get()
            with end_condition:
                with start_condition:
                    start_condition.notify()
                end_condition.wait()

    def put(
        self,
        traj: Sequence[PPOTransition],
        next_obs: Union[Observation, ObservationGlobalState],
        next_dones: Array,
        time_dict: Dict,
    ) -> None:
        """
        Put a trajectory on the queue to be consumed by the learner.
        """
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()  # wait to be allowed to start

        # [PPOTransition()] * rollout_len --> PPOTransition[done=(rollout_len, num_envs, num_agents)
        sharded_traj = jax.tree.map(lambda *x: self.shard_split_playload(jnp.stack(x), 1), *traj)

        # obs Tuple[(num_envs, num_agents, ...), ...] -->
        # [(num_envs / num_learner_devices, num_agents, ...)] * num_learner_devices
        sharded_next_obs = jax.tree.map(self.shard_split_playload, next_obs)

        # dones (num_envs, num_agents) -->
        # [(num_envs / num_learner_devices, num_agents)] * num_learner_devices
        sharded_next_dones = self.shard_split_playload(next_dones, 0)

        self._queue.put((sharded_traj, sharded_next_obs, sharded_next_dones, time_dict))

        with end_condition:
            end_condition.notify()  # tell we have finish

    def qsize(self) -> int:
        """Returns the number of trajectories in the pipeline."""
        return self._queue.qsize()

    def get(
        self, block: bool = True, timeout: Union[float, None] = None
    ) -> Tuple[PPOTransition, Union[Observation, ObservationGlobalState], Array, Dict]:
        """Get a trajectory from the pipeline."""
        return self._queue.get(block, timeout)  # type: ignore

    def shard_split_playload(self, payload: Any, axis: int = 0) -> Any:
        split_payload = jnp.split(payload, len(self.learner_devices), axis=axis)
        return jax.device_put_sharded(split_payload, devices=self.learner_devices)


class ParamsSource(threading.Thread):
    """
    A `ParamSource` is a component that allows networks params to be passed from a
    `Learner` component to `Actor` components.
    """

    def __init__(self, init_value: Params, device: jax.Device):
        super().__init__(name=f"ParamsSource-{device.id}")
        self.value: Params = jax.device_put(init_value, device)
        self.device = device
        self.new_value: queue.Queue = queue.Queue()

    def run(self) -> None:
        """
        This function is responsible for updating the value of the `ParamSource` when a new value
        is available.
        """
        while True:
            try:
                waiting = self.new_value.get(block=True, timeout=1)
                self.value = jax.device_put(jax.block_until_ready(waiting), self.device)
            except queue.Empty:
                continue

    def update(self, new_params: Params) -> None:
        """
        Update the value of the `ParamSource` with a new value.

        Args:
            new_params: The new value to update the `ParamSource` with.
        """
        self.new_value.put(new_params)

    def get(self) -> Params:
        """Get the current value of the `ParamSource`."""
        return self.value


class RecordTimeTo:
    def __init__(self, to: Any):
        self.to = to

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, *args: Any) -> None:
        end = time.monotonic()
        self.to.append(end - self.start)


class ThreadLifetime:
    """Simple class for a mutable boolean that can be used to signal a thread to stop."""

    def __init__(self) -> None:
        self._stop = False

    def should_stop(self) -> bool:
        return self._stop

    def stop(self) -> None:
        self._stop = True
