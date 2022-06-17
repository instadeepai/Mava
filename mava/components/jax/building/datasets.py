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
from types import SimpleNamespace
from typing import Any, Callable, Optional

import reverb
from acme import datasets

from mava.components.jax import Component
from mava.core_jax import SystemBuilder

Transform = Callable[[reverb.ReplaySample], reverb.ReplaySample]


class TrainerDataset(Component):
    @abc.abstractmethod
    def __init__(
        self,
        local_config: Any,
        global_config: SimpleNamespace = SimpleNamespace(),
    ):
        """_summary_

        Args:
            local_config : _description_.
            global_config : _description_.
        """
        self.local_config = local_config
        self.global_config = global_config

    @abc.abstractmethod
    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "trainer_dataset"


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
        local_config: TransitionDatasetConfig = TransitionDatasetConfig(),
        global_config: SimpleNamespace = SimpleNamespace(),
    ):
        """_summary_

        Args:
            local_config : _description_.
            global_config : _description_.
        """
        self.local_config = local_config
        self.global_config = global_config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        max_in_flight_samples_per_worker = self.local_config.max_in_flight_samples_per_worker
        dataset = datasets.make_reverb_dataset(
            table=builder.store.trainer_id,
            server_address=builder.store.data_server_client.server_address,
            batch_size=self.local_config.sample_batch_size,
            prefetch_size=self.local_config.prefetch_size,
            num_parallel_calls=self.local_config.num_parallel_calls,
            max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
            postprocess=self.local_config.postprocess,
        )

        builder.store.dataset = iter(dataset)

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return TransitionDatasetConfig


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
        local_config: TrajectoryDatasetConfig = TrajectoryDatasetConfig(),
        global_config: SimpleNamespace = SimpleNamespace(),
    ):
        """_summary_

        Args:
            local_config : _description_.
            global_config : _description_.
        """
        self.local_config = local_config
        self.global_config = global_config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=builder.store.data_server_client.server_address,
            table=builder.store.trainer_id,
            max_in_flight_samples_per_worker=2 * self.local_config.sample_batch_size,
            num_workers_per_iterator=self.local_config.num_workers_per_iterator,
            max_samples_per_stream=self.local_config.max_samples_per_stream,
            rate_limiter_timeout_ms=self.local_config.rate_limiter_timeout_ms,
            get_signature_timeout_secs=self.local_config.get_signature_timeout_secs,
            # max_samples=self.config.max_samples,
        )

        # Add batch dimension.
        dataset = dataset.batch(self.local_config.sample_batch_size, drop_remainder=True)
        builder.store.sample_batch_size = self.local_config.sample_batch_size

        builder.store.dataset_iterator = dataset.as_numpy_iterator()

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return TrajectoryDatasetConfig
