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

"""Commonly used adder components for system builders"""
import abc

from typing import Dict, Any, List

from mava import specs
from mava.callbacks import Callback
from mava.core import SystemBuilder
from mava.adders import reverb as reverb_adders


class Adder(Callback):
    @abc.abstractmethod
    def on_building_adder_make_adder(self, builder: SystemBuilder) -> None:
        """[summary]"""


class AdderSignature(Callback):
    @abc.abstractmethod
    def on_building_tables_adder_signature(self, builder: SystemBuilder) -> None:
        """[summary]"""


class ParallelNStepTransitionAdder(Adder):
    def __init__(
        self,
        n_step: int,
        discount: float,
    ):

        self.n_step = n_step
        self.discount = discount

    def on_building_adder_make_adder(self, builder: SystemBuilder) -> None:
        adder = reverb_adders.ParallelNStepTransitionAdder(
            priority_fns=builder.priority_fns,
            client=builder._replay_client,
            net_ids_to_keys=builder.unique_net_keys,
            n_step=self.n_step,
            table_network_config=builder.table_network_config,
            discount=self.discount,
        )

        builder.adder = adder


class ParallelNStepTransitionAdderSignature(AdderSignature):
    def on_building_tables_adder_signature(self, builder: SystemBuilder) -> None:
        def adder_sig_fn(
            env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
        ) -> Any:
            return reverb_adders.ParallelNStepTransitionAdder.signature(
                env_spec, extra_specs
            )

        builder.adder_signature_fn = adder_sig_fn


class ParallelSequenceAdder(Adder):
    def __init__(
        self,
        sequence_length: int,
        period: int,
    ):

        self.sequence_length = sequence_length
        self.period = period

    def on_building_adder_make_adder(self, builder: SystemBuilder) -> None:
        adder = reverb_adders.ParallelNStepTransitionAdder(
            priority_fns=builder.priority_fns,
            client=builder._replay_client,
            net_ids_to_keys=builder.unique_net_keys,
            sequence_length=self.sequence_length,
            table_network_config=builder.table_network_config,
            period=self.period,
        )

        builder.adder = adder


class ParallelSequenceAdderSignature(AdderSignature):
    def __init__(self, sequence_length: int = 20) -> None:
        """[summary]

        Args:
            sequence_length (int, optional): [description]. Defaults to 20.
        """
        self.sequence_length = sequence_length

    def on_building_tables_adder_signature(self, builder: SystemBuilder) -> None:
        def adder_sig_fn(
            env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
        ) -> Any:
            return reverb_adders.ParallelSequenceAdder.signature(
                env_spec, self.sequence_length, extra_specs
            )

        builder.adder_signature = adder_sig_fn
