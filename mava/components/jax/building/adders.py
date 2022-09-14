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
from types import SimpleNamespace
from typing import Any, Dict, List, Type

from mava import specs
from mava.adders import reverb as reverb_adders
from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.building.system_init import BaseSystemInit
from mava.components.jax.training.trainer import BaseTrainerInit
from mava.core_jax import SystemBuilder


class Adder(Component):
    """Abstract Adder component defining which hooks should be used."""

    @abc.abstractmethod
    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """Create the executor adder.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_adder"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseTrainerInit required to set up builder.store.table_network_config.
        BaseSystemInit required to set up builder.store.unique_net_keys.

        Returns:
            List of required component classes.
        """
        return [BaseTrainerInit, BaseSystemInit]


class AdderPriority(Component):
    """Abstract AdderPriority component defining which hooks should be used."""

    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ):
        """Component creates priority functions for reverb adders.

        Args:
            config: SimpleNamespace.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """Create the priority functions.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "adder_priority"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseTrainerInit required to set up builder.store.table_network_config.

        Returns:
            List of required component classes.
        """
        return [BaseTrainerInit]


class AdderSignature(Component):
    """Abstract AdderSignature component defining which hooks should be used."""

    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ):
        """Component creates an adder signature for reverb adders.

        Args:
            config: SimpleNamespace.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """Create the adder signature function.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "data_server_adder_signature"


@dataclass
class ParallelTransitionAdderConfig:
    n_step: int = 5
    discount: float = 0.99


class ParallelTransitionAdder(Adder):
    def __init__(
        self,
        config: ParallelTransitionAdderConfig = ParallelTransitionAdderConfig(),
    ):
        """Creates a reverb ParallelNStepTransitionAdder.

        Args:
            config: ParallelTransitionAdderConfig.
        """
        self.config = config

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """Create a ParallelNStepTransitionAdder.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """

        adder = reverb_adders.ParallelNStepTransitionAdder(
            priority_fns=builder.store.priority_fns,
            client=builder.store.data_server_client,  # Created by builder
            net_ids_to_keys=builder.store.unique_net_keys,
            n_step=self.config.n_step,
            table_network_config=builder.store.table_network_config,
            discount=self.config.discount,
        )

        builder.store.adder = adder


class UniformAdderPriority(AdderPriority):
    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """Create and store the adder priority functions.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        builder.store.priority_fns = {
            table_key: lambda x: 1.0
            for table_key in builder.store.table_network_config.keys()
        }


class ParallelTransitionAdderSignature(AdderSignature):
    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """Create a ParallelNStepTransitionAdder signature.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """

        def adder_sig_fn(
            ma_environment_spec: specs.MAEnvironmentSpec,
            extras_specs: Dict[str, Any],
        ) -> Any:
            """Constructs a ParallelNStepTransitionAdder signature from specs.

            Args:
                ma_environment_spec: Environment specs.
                extras_specs: Other specs.

            Returns:
                ParallelNStepTransitionAdder signature.
            """
            return reverb_adders.ParallelNStepTransitionAdder.signature(
                ma_environment_spec=ma_environment_spec, extras_specs=extras_specs
            )

        builder.store.adder_signature_fn = adder_sig_fn


@dataclass
class ParallelSequenceAdderConfig:
    sequence_length: int = 20
    period: int = 10
    use_next_extras: bool = False


class ParallelSequenceAdder(Adder):
    def __init__(
        self, config: ParallelSequenceAdderConfig = ParallelSequenceAdderConfig()
    ):
        """Component creates a reverb ParallelSequenceAdder.

        Args:
            config: ParallelSequenceAdderConfig.
        """
        self.config = config

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """Create a ParallelSequenceAdder.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """

        adder = reverb_adders.ParallelSequenceAdder(
            priority_fns=builder.store.priority_fns,
            client=builder.store.data_server_client,  # Created by builder
            net_ids_to_keys=builder.store.unique_net_keys,
            sequence_length=self.config.sequence_length,
            table_network_config=builder.store.table_network_config,
            period=self.config.period,
            use_next_extras=self.config.use_next_extras,
        )

        builder.store.adder = adder


class ParallelSequenceAdderSignature(AdderSignature):
    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """Create a ParallelSequenceAdder signature.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """

        def adder_sig_fn(
            ma_environment_spec: specs.MAEnvironmentSpec,
            sequence_length: int,
            extras_specs: Dict[str, Any],
        ) -> Any:
            """Creates a ParallelSequenceAdder signature.

            Args:
                ma_environment_spec: Environment specs.
                sequence_length: Length of the adder sequences.
                extras_specs: Other specs.

            Returns:
                ParallelSequenceAdder signature.
            """
            return reverb_adders.ParallelSequenceAdder.signature(
                ma_environment_spec=ma_environment_spec,
                sequence_length=sequence_length,
                extras_specs=extras_specs,
            )

        builder.store.adder_signature_fn = adder_sig_fn
