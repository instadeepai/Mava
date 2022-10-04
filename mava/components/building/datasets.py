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

"""Commonly used dataset components for system builders"""
import abc
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Type

import reverb
from acme import datasets

from mava.callbacks import Callback
from mava.components import Component
from mava.core_jax import SystemBuilder

Transform = Callable[[reverb.ReplaySample], reverb.ReplaySample]


class TrainerDataset(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: Any,
    ):
        """Component creates an iterable dataset from existing reverb table.

        Args:
            config: Any.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """Abstract method defining hook to be overridden.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "trainer_dataset"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []


@dataclass
class TransitionDatasetConfig:
    sample_batch_size: int = 256
    prefetch_size: Optional[int] = None
    num_parallel_calls: int = 12
    max_in_flight_samples_per_worker: Optional[int] = None
    postprocess: Optional[Transform] = None
    # dataset_name: str = "transition_dataset"


class TransitionDataset(TrainerDataset):
    def __init__(
        self,
        config: TransitionDatasetConfig = TransitionDatasetConfig(),
    ):
        """Component creates a reverb transition dataset for the trainer.

        Args:
            config: TransitionDatasetConfig.
        """
        self.config = config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """Build a transition dataset and save it to the store.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        max_in_flight_samples_per_worker = self.config.max_in_flight_samples_per_worker
        dataset = datasets.make_reverb_dataset(
            table=builder.store.trainer_id,  # Set by builder
            # Set by builder
            server_address=builder.store.data_server_client.server_address,
            batch_size=self.config.sample_batch_size,
            prefetch_size=self.config.prefetch_size,
            num_parallel_calls=self.config.num_parallel_calls,
            max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
            postprocess=self.config.postprocess,
        )

        builder.store.dataset_iterator = iter(dataset)


@dataclass
class TrajectoryDatasetConfig:
    sample_batch_size: int = 256
    max_in_flight_samples_per_worker: int = 512
    num_workers_per_iterator: int = -1
    max_samples_per_stream: int = -1
    rate_limiter_timeout_ms: int = -1
    get_signature_timeout_secs: Optional[int] = None
    # max_samples: int = -1
    # dataset_name: str = "trajectory_dataset"


class TrajectoryDataset(TrainerDataset):
    def __init__(
        self,
        config: TrajectoryDatasetConfig = TrajectoryDatasetConfig(),
    ):
        """Component creates a reverb trajectory dataset for the trainer.

        Args:
            config: TrajectoryDatasetConfig.
        """
        self.config = config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """Build a trajectory dataset and save it to the store.

        Automatically adds a batch dimension to the dataset.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=builder.store.data_server_client.server_address,
            table=builder.store.trainer_id,
            max_in_flight_samples_per_worker=2 * self.config.sample_batch_size,
            num_workers_per_iterator=self.config.num_workers_per_iterator,
            max_samples_per_stream=self.config.max_samples_per_stream,
            rate_limiter_timeout_ms=self.config.rate_limiter_timeout_ms,
            get_signature_timeout_secs=self.config.get_signature_timeout_secs,
            # max_samples=self.config.max_samples,
        )

        # Add batch dimension.
        dataset = dataset.batch(self.config.sample_batch_size, drop_remainder=True)

        builder.store.dataset_iterator = dataset.as_numpy_iterator()
