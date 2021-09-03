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

"""Commonly used rate limiter components for system builders"""
import reverb

from mava import specs
from mava.callbacks import Callback
from mava.systems.building import BaseSystemBuilder
from mava.adders import reverb as reverb_adders


class RateLimiter(Callback):
    def __init__(
        self,
        samples_per_insert: Optional[float] = 32.0,
        min_replay_size: int = 1000,
    ) -> None:
        """[summary]

        Args:
            samples_per_insert (Optional[float], optional): [description].
            min_replay_size (int, optional): [description].
        """
        self.sample_per_insert = samples_per_insert
        self.min_replay_size = min_replay_size

    def on_building_rate_limiter(self, builder: BaseSystemBuilder) -> None:
        """[summary]

        Args:
            builder (BaseSystemBuilder): [description]
        """
        pass


class OffPolicyRateLimiter(RateLimiter):
    def on_building_rate_limiter(self, builder: BaseSystemBuilder) -> None:
        if self.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            def limiter_fn() -> reverb.rate_limiters:
                return reverb.rate_limiters.MinSize(self.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self.samples_per_insert
            error_buffer = self.min_replay_size * samples_per_insert_tolerance

            def limiter_fn() -> reverb.rate_limiters:
                return reverb.rate_limiters.SampleToInsertRatio(
                    min_size_to_sample=self.min_replay_size,
                    samples_per_insert=self.samples_per_insert,
                    error_buffer=error_buffer,
                )

        builder.rate_limiter_fn = limiter_fn