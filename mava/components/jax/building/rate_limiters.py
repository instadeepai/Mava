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
import abc
from dataclasses import dataclass
from typing import Callable, List, Optional, Type

import reverb

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.core_jax import SystemBuilder


@dataclass
class RateLimiterConfig:
    min_data_server_size: int = 1000
    samples_per_insert: float = 32.0
    error_buffer: Optional[float] = None


class RateLimiter(Component):
    def __init__(self, config: RateLimiterConfig = RateLimiterConfig()) -> None:
        """[summary]"""
        self.config = config

    @abc.abstractmethod
    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """[summary]"""

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "rate_limiter"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return RateLimiterConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []


class MinSizeRateLimiter(RateLimiter):
    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """

        def rate_limiter_fn() -> reverb.rate_limiters:
            return reverb.rate_limiters.MinSize(self.config.min_data_server_size)

        builder.store.rate_limiter_fn = rate_limiter_fn


class SampleToInsertRateLimiter(RateLimiter):
    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        Returns:
            _description_
        """
        if not self.config.error_buffer:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self.config.samples_per_insert
            error_buffer = (
                self.config.min_data_server_size * samples_per_insert_tolerance
            )
        else:
            error_buffer = self.config.error_buffer

        def rate_limiter_fn() -> reverb.rate_limiters:
            return reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self.config.min_data_server_size,
                samples_per_insert=self.config.samples_per_insert,
                error_buffer=error_buffer,
            )

        builder.store.rate_limiter_fn = rate_limiter_fn
