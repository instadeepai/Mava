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

import threading
from math import ceil
from typing import Optional, Tuple, Union

from colorama import Fore, Style


# from https://github.com/EdanToledo/Stoix/blob/feat/sebulba-dqn/stoix/utils/rate_limiters.py
class RateLimiter:
    """
    Rate limiter to control the ratio of samples to inserts.

    This class is designed to regulate the rate at which samples are drawn
    compared to the rate at which new data is inserted.

    Args:
        samples_per_insert (float): The target ratio of samples to inserts.
            For example, a value of 4.0 means the system aims to sample 4 times
            for every insertion.
        min_size_to_sample (int): The minimum number of inserts required before
            sampling is allowed.
        min_diff (float): The minimum acceptable difference between the expected
            number of samples (based on inserts and `samples_per_insert`) and
            the actual number of samples. Sampling is allowed if the difference
            is greater than or equal to this value.
        max_diff (float): The maximum acceptable difference between the expected
            number of samples and the actual number of samples. Inserting is allowed
            if the difference is less than or equal to this value.
    """

    def __init__(
        self, samples_per_insert: float, min_size_to_sample: float, min_diff: float, max_diff: float
    ):
        assert min_size_to_sample > 0, "min_size_to_sample must be greater than 0"
        assert samples_per_insert > 0, "samples_per_insert must be greater than 0"

        self.samples_per_insert = samples_per_insert
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.min_size_to_sample = min_size_to_sample

        self.inserts = 0.0
        self.samples = 0

        self.mutex = threading.Lock()
        self.condition = threading.Condition(self.mutex)

    def num_inserts(self) -> float:
        """Returns the number of inserts."""
        with self.mutex:
            return self.inserts

    def num_samples(self) -> int:
        """Returns the number of samples."""
        with self.mutex:
            return self.samples

    def insert(self, insert_fraction: float = 1) -> None:
        """Increment the number of inserts and notify all waiting threads."""
        with self.mutex:
            self.inserts += insert_fraction
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
        if ceil(self.inserts) + num_inserts <= self.min_size_to_sample:
            return True
        diff = (num_inserts + ceil(self.inserts)) * self.samples_per_insert - self.samples
        return diff <= self.max_diff

    def can_sample(self, num_samples: int) -> bool:
        """Check if the caller can sample `num_samples` items."""
        # Assume lock is already held by the caller
        if num_samples <= 0:
            return False
        if ceil(self.inserts) < self.min_size_to_sample:
            return False
        diff = ceil(self.inserts) * self.samples_per_insert - self.samples - num_samples
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
        if isinstance(error_buffer, (int, float)):
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


class BlockingRatioLimiter(RateLimiter):
    """
    Blocking rate limiter that enforces a ratio of X samples per insert and 1/X inserts per sample.

    Args:
        sample_insert_ratio (float): The ratio of samples to inserts (X).
            For example, a value of 2.0 means for every insert, up to 2 samples are allowed,
            and for every 2 samples, up to 1 insert is allowed. Must be greater than 0.
    """

    def __init__(self, sample_insert_ratio: float, min_num_inserts: float):
        if sample_insert_ratio <= 0:
            raise ValueError("sample_insert_ratio must be greater than 0")
        super().__init__(
            samples_per_insert=sample_insert_ratio,
            min_size_to_sample=min_num_inserts,
            min_diff=float("-inf"),
            max_diff=float("inf"),
        )
        self.available_inserts = 1.0
        self.available_samples = 0.0
        self.sample_insert_ratio = sample_insert_ratio

    def insert(self, insert_fraction: float = 1.0) -> None:
        """
        Increments the available samples by insert_fraction * sample_insert_ratio.
        """
        with self.mutex:
            if self.min_size_to_sample > 0:
                self.min_size_to_sample -= insert_fraction
            else:
                self.available_samples += insert_fraction * self.sample_insert_ratio
                self.available_inserts -= insert_fraction

            self.inserts += insert_fraction
            self.condition.notify_all()

    def sample(self, num_samples: int = 1) -> None:
        """
        Increments the available inserts by num_samples / sample_insert_ratio.
        """
        with self.mutex:
            self.available_inserts += num_samples / self.sample_insert_ratio
            self.available_samples -= num_samples
            self.samples += 1
            self.condition.notify_all()

    def can_insert(self, num_inserts: float = 1.0) -> bool:
        """
        Checks if it is possible to insert num_inserts, based on available insert credits.
        """
        return self.available_inserts >= num_inserts

    def can_sample(self, num_samples: int = 1) -> bool:
        """
        Checks if it is possible to sample num_samples, based on available sample credits.
        """
        return self.available_samples >= num_samples
