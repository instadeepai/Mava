
import queue
import threading
from typing import Any, Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from flashbax import make_trajectory_buffer
from jax import tree
from jax.sharding import Sharding
from jumanji.types import TimeStep
from omegaconf import DictConfig

from mava.utils.sebulba.rate_limiters import RateLimiter
from mava.systems.ppo.types import PPOTransition
from mava.systems.q_learning.types import Transition
from mava.types import Metrics

QUEUE_PUT_TIMEOUT = 100

@jax.jit
def _stack_trajectory(
    trajectory: Union[List[PPOTransition], List[Transition]],
) -> Union[PPOTransition, Transition]:
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
        self._stop_event = threading.Event()

    def run(self) -> None:
        """This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self._stop_event.is_set():
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

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()


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
        init_transition: Transition,
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
        self._timing_queue: queue.Queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

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
        self.buffer_adds_count = [0] * self.num_buffers

        # Setup functions
        self.buffer_add = jax.jit(rb.add, device=self.cpu)
        self.buffer_sample = jax.jit(rb.sample, device=self.cpu)

    def run(self) -> None:
        """This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self._stop_event.is_set():
            try:
                start_condition, end_condition = self.tickets_queue.get(timeout=1)
                with end_condition:
                    with start_condition:
                        start_condition.notify()
                    end_condition.wait()
            except queue.Empty:
                continue

    def put(self, traj: Sequence[Transition], metrics: Tuple, actor_id: int) -> None:
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
        # [{'metric1' : value1, ...} * rollout_len -> {'metric1' : [value1, value2, ...], ...}
        episode_metrics = _stack_trajectory(episode_metrics)

        self.buffer_states[actor_id] = self.buffer_add(self.buffer_states[actor_id], traj)
        self.buffer_adds_count[actor_id] += 1

        if self._timing_queue.full():
            self._timing_queue.get() # remove the oldest entry

        self._timing_queue.put((time_dict, episode_metrics))

        self.rate_limiter.insert(1 / self.num_buffers)

        with end_condition:
            end_condition.notify()  # notify that we have finished

    def get(self, timeout: Union[float, None] = None) -> Tuple[Transition, Any]:
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
        sampled_batch: List[Transition] = [
            self.buffer_sample(state, sample_key).experience for state in self.buffer_states
        ]
        transitions: Transition = tree.map(lambda *x: np.concatenate(x), *sampled_batch)
        transitions = jax.device_put(transitions, device=self.sharding)

        self.rate_limiter.sample()

        if not self._timing_queue.empty():
            return transitions, self._timing_queue.get()

        return transitions, (None, None)

    def clear(self) -> None:
        """Clear the pipeline."""
        while not self._timing_queue.empty():
            try:
                self._timing_queue.get(block=False)
            except queue.Empty:
                break

    def qsize(self) -> int:
        """Returns the number of trajectories in the pipeline."""
        return self._timing_queue.qsize()
    
    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()