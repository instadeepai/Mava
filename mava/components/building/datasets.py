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

from acme import datasets

from mava.callbacks import Callback
from mava.core import SystemBuilder


class Iterator(Callback):
    def __init__(
        self,
        name: str = "data_store",
        batch_size: Optional[int] = None,
        prefetch_size: Optional[int] = None,
        num_parallel_calls: int = 12,
        max_in_flight_samples_per_worker: Optional[int] = None,
    ) -> None:
        """[summary]

        Args:
            server_address (str): [description]
            batch_size (Optional[int], optional): [description]. Defaults to None.
            prefetch_size (Optional[int], optional): [description]. Defaults to None.
            table (str, optional): [description]. Defaults to adders.DEFAULT_PRIORITY_TABLE.
            num_parallel_calls (int, optional): [description]. Defaults to 12.
            max_in_flight_samples_per_worker (Optional[int], optional): [description]. Defaults to None.
        """
        self.table_name = name
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.num_parallel_calls = num_parallel_calls
        self.max_in_flight_samples_per_worker = max_in_flight_samples_per_worker

    @abc.abstractmethod
    def on_building_dataset_make_dataset(self, builder: SystemBuilder) -> None:
        """[summary]"""


class Dataset(Iterator):
    def on_building_dataset_make_dataset(self, builder: SystemBuilder) -> None:
        dataset = datasets.make_reverb_dataset(
            table=self.table_name,
            server_address=builder.replay_client.server_address,
            batch_size=self.batch_size,
            prefetch_size=self.prefetch_size,
            num_parallel_calls=self.num_parallel_calls,
            max_in_flight_samples_per_worker=self.max_in_flight_samples_per_worker,
        )

        builder.dataset = iter(dataset)
