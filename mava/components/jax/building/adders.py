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
from dataclasses import dataclass
from typing import Any, Dict

from mava import specs
from mava.adders import reverb as reverb_adders
from mava.callbacks import Callback
from mava.core_jax import SystemBuilder


class Adder(Callback):
    @abc.abstractmethod
    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """[summary]"""


class AdderSignature(Callback):
    @abc.abstractmethod
    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """[summary]"""


@dataclass
class ParallelTransitionAdderConfig:
    n_step: int = 5
    discount: float = 0.99


class ParallelTransitionAdder(Adder):
    def __init__(
        self,
        config: ParallelTransitionAdderConfig = ParallelTransitionAdderConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        adder = reverb_adders.ParallelNStepTransitionAdder(
            priority_fns=builder.attr.priority_fns,
            client=builder.attr.system_data_server,
            net_ids_to_keys=builder.attr.unique_net_keys,
            n_step=self.config.n_step,
            table_network_config=builder.attr.table_network_config,
            discount=self.config.discount,
        )

        builder.attr.adder = adder


class ParallelTransitionAdderSignature(AdderSignature):
    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """

        def adder_sig_fn(
            env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
        ) -> Any:
            return reverb_adders.ParallelNStepTransitionAdder.signature(
                env_spec, extra_specs
            )

        builder.attr.adder_signature_fn = adder_sig_fn


@dataclass
class ParallelSequenceAdderConfig:
    sequence_length: int = 20
    period: int = 20


class ParallelSequenceAdder(Adder):
    def __init__(
        self, config: ParallelSequenceAdderConfig = ParallelSequenceAdderConfig()
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.attr.sequence_length = self.config.sequence_length

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        adder = reverb_adders.ParallelSequenceAdder(
            priority_fns=builder.attr.priority_fns,
            client=builder.attr.system_data_server,
            net_ids_to_keys=builder.attr.unique_net_keys,
            sequence_length=self.config.sequence_length,
            table_network_config=builder.attr.table_network_config,
            period=self.config.period,
        )

        builder.attr.adder = adder


class ParallelSequenceAdderSignature(AdderSignature):
    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """

        def adder_sig_fn(
            environment_spec: specs.MAEnvironmentSpec,
            sequence_length: int,
            extras_spec: Dict[str, Any],
        ) -> Any:
            return reverb_adders.ParallelSequenceAdder.signature(
                environment_spec=environment_spec,
                sequence_length=sequence_length,
                extras_spec=extras_spec,
            )

        builder.attr.adder_signature_fn = adder_sig_fn
