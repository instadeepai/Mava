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

"""Commonly used rate limiter, sampler and remover components for system builders"""
import abc
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Type

import reverb

from mava.callbacks import Callback
from mava.components import Component
from mava.core_jax import SystemBuilder


@dataclass
class RateLimiterConfig:
    min_data_server_size: int = 1000
    samples_per_insert: float = 32.0
    error_buffer: Optional[float] = None


class RateLimiter(Component):
    def __init__(self, config: RateLimiterConfig = RateLimiterConfig()) -> None:
        """Creates reverb rate limiter functions"""
        self.config = config

    @abc.abstractmethod
    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """Hook for adding reverb rate limiter function to builder store."""

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "rate_limiter"

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
        """Block sample calls unless replay contains `min_size_to_sample`.

        This limiter blocks all sample calls when the replay contains less than
        `min_size_to_sample` items, and accepts all sample calls otherwise.

        Args:
            builder : system builder
        """

        def rate_limiter_fn() -> reverb.rate_limiters:
            """Function to retrieve rate limiter."""
            return reverb.rate_limiters.MinSize(self.config.min_data_server_size)

        builder.store.rate_limiter_fn = rate_limiter_fn


class SampleToInsertRateLimiter(RateLimiter):
    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """Maintains a specified ratio between samples and inserts.

        The limiter works in two stages:

            Stage 1. Size of table is lt `min_size_to_sample`.
            Stage 2. Size of table is ge `min_size_to_sample`.

        During stage 1 the limiter works exactly like MinSize, i.e. it allows
        all insert calls and blocks all sample calls. Note that it is possible to
        transition into stage 1 from stage 2 when items are removed from the table.

        During stage 2 the limiter attempts to maintain the ratio
        `samples_per_inserts` between the samples and inserts. This is done by
        measuring the "error" in this ratio, calculated as:

            number_of_inserts * samples_per_insert - number_of_samples

        If `error_buffer` is a number and this quantity is larger than
        `min_size_to_sample * samples_per_insert + error_buffer` then insert calls
        will be blocked; sampling will be blocked for error less than
        `min_size_to_sample * samples_per_insert - error_buffer`.

        If `error_buffer` is a tuple of two numbers then insert calls will block if
        the error is larger than error_buffer[1], and sampling will block if the error
        is less than error_buffer[0].

        `error_buffer` exists to avoid unnecessary blocking for a system that is
        more or less in equilibrium.

        Args:
            builder : system builder
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
            """Function to retrieve rate limiter."""
            return reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self.config.min_data_server_size,
                samples_per_insert=self.config.samples_per_insert,
                error_buffer=error_buffer,
            )

        builder.store.rate_limiter_fn = rate_limiter_fn


class Sampler(Component):
    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ):
        """Creates reverb selector functions for sampling data.

        These functions dictate how experience will be sampled form the
        replay table.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """Hook for adding reverb selector to builder store.

        This determines how experience will be sampled from the replay table.
        """

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""

        return "data_server_sampler"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []


class UniformSampler(Sampler):
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """Sample data from the table uniformly"""

        def sampler_fn() -> reverb.selectors:
            """Function to retrieve uniform reverb sampler."""
            return reverb.selectors.Uniform()

        builder.store.sampler_fn = sampler_fn


class MinHeapSampler(Sampler):
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """Sample data from the table with min heap"""

        def sampler_fn() -> reverb.selectors:
            return reverb.selectors.MinHeap()

        builder.store.sampler_fn = sampler_fn


@dataclass
class PrioritySamplerConfig:
    priority_exponent: float = 1.0


class PrioritySampler(Sampler):
    def __init__(
        self,
        config: PrioritySamplerConfig = PrioritySamplerConfig(),
    ):
        """Initialise priortized sampling from replay table."""

        self.config = config

    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """Sample experience from the replay table according to its importance."""

        def sampler_fn() -> reverb.selectors:
            """Function to retrieve sampler."""
            return reverb.selectors.Prioritized(self.config.priority_exponent)

        builder.store.sampler_fn = sampler_fn


class Remover(Component):
    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ):
        """Creates reverb selector functions for removing data.

        These functions dictate how experience will be removed form the
        replay table once the maximum replay table size is reached.

        Args:
            config: RemoverConfig.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """Hook for adding reverb selector to builder store.

        This determines how experience will be sampled from the replay table.

        Args:
            builder: SystemBuilder.

        Returns:
            config class/dataclass for component.
        """

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""

        return "data_server_remover"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []


class FIFORemover(Remover):
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """First In First Out remover.

        The experience that was added to the replay table earlier is removed first.

        Args:
            builder: SystemBuilder.

        Returns:
            config class/dataclass for component.
        """

        def remover_fn() -> reverb.selectors:
            return reverb.selectors.Fifo()

        builder.store.remover_fn = remover_fn


class LIFORemover(Remover):
    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """Last In First Out remover.

        The experience that was added to the replay table later is removed first.

        Args:
            builder: SystemBuilder.

        Returns:
            config class/dataclass for component.
        """

        def remover_fn() -> reverb.selectors:
            return reverb.selectors.Lifo()

        builder.store.remover_fn = remover_fn
