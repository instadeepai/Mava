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
from typing import Optional

import reverb
from acme import datasets

from mava.callbacks import Callback
from mava.core import SystemBuilder


class Iterator(Callback):
    def __init__(
        self,
        batch_size: Optional[int] = None,
        sequence_length=None,
        prefetch_size: Optional[int] = None,
        num_parallel_calls: int = 12,
        max_in_flight_samples_per_worker: Optional[int] = None,
    ) -> None:
        """[summary]

        Args:
            server_address: [description].
            batch_size: [description].
            sequence_length: [description].
            prefetch_size: [description].
            num_parallel_calls: [description].
            max_in_flight_samples_per_worker: [description].
        """
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._prefetch_size = prefetch_size
        self._num_parallel_calls = num_parallel_calls
        self._max_in_flight_samples_per_worker = max_in_flight_samples_per_worker

    @abc.abstractmethod
    def on_building_dataset_make_dataset(self, builder: SystemBuilder) -> None:
        """[summary]"""


class AcmeDataset(Iterator):
    def on_building_dataset_make_dataset(self, builder: SystemBuilder) -> None:
        dataset = datasets.make_reverb_dataset(
            table=builder._table_name,
            server_address=builder._replay_client.server_address,
            batch_size=self._batch_size,
            prefetch_size=self._prefetch_size,
            num_parallel_calls=self._num_parallel_calls,
            max_in_flight_samples_per_worker=self._max_in_flight_samples_per_worker,
        )

        builder.dataset = iter(dataset)


class ReverbDataset(Iterator):
    def on_building_dataset_make_dataset(self, builder: SystemBuilder) -> None:
        """Create a dataset iterator to use for learning/updating the system."""
        # New dataset
        dataset = reverb.dataset.ReplayDataset.from_table_signature(
            table=builder._table_name,
            server_address=builder._replay_client.server_address,
            sequence_length=self._sequence_length,
            max_in_flight_samples_per_worker=self._max_in_flight_samples_per_worker,
            emit_timesteps=False,
            rate_limiter_timeout_ms=10,
        ).batch(self._batch_size)
        builder.dataset = iter(dataset)
