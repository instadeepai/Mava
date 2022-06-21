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
from typing import Any, Callable, Dict

from mava import specs
from mava.adders import reverb as reverb_adders
from mava.components.jax import Component
from mava.core_jax import SystemBuilder


class Adder(Component):
    @abc.abstractmethod
    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """[summary]"""

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "adder"


@dataclass
class AdderPriorityConfig:
    pass


class AdderPriority(Component):
    def __init__(
        self,
        config: AdderPriorityConfig = AdderPriorityConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        Returns:
            _description_
        """

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "adder_priority"

    @staticmethod
    def config_class() -> Callable:
        """Returns the class of the config for this component."""
        return AdderPriorityConfig


@dataclass
class AdderSignatureConfig:
    pass


class AdderSignature(Component):
    def __init__(
        self,
        config: AdderSignatureConfig = AdderSignatureConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """[summary]"""

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "adder_signature"

    @staticmethod
    def config_class() -> Callable:
        """Returns the class of the config for this component."""
        return AdderSignatureConfig


@dataclass
class ParallelTransitionAdderConfig:
    n_step: int = 1
    discount: float = 0.99
    # use_next_extras: bool = False


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
        if not hasattr(builder.store, "priority_fns"):
            builder.store.priority_fns = None

        adder = reverb_adders.ParallelNStepTransitionAdder(
            priority_fns=builder.store.priority_fns,
            client=builder.store.data_server_client,
            net_ids_to_keys=builder.store.unique_net_keys,
            n_step=self.config.n_step,
            table_network_config=builder.store.table_network_config,
            discount=self.config.discount,
            # use_next_extras=self.config.use_next_extras,
        )

        builder.store.adder = adder

    @staticmethod
    def config_class() -> Callable:
        """Returns the class of the config for this component."""
        return ParallelTransitionAdderConfig

    @staticmethod
    def name() -> str:
        """Returns the name of the component"""
        return "parallel_transition_adder"


class UniformAdderPriority(AdderPriority):
    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.store.priority_fns = {
            table_key: lambda x: 1.0
            for table_key in builder.store.table_network_config.keys()
        }


class ParallelTransitionAdderSignature(AdderSignature):
    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """

        def adder_sig_fn(
            env_spec: specs.MAEnvironmentSpec,
            extra_specs: Dict[str, Any],
            next_extra_specs: Dict[str, Any],
        ) -> Any:
            return reverb_adders.ParallelNStepTransitionAdder.signature(
                environment_spec=env_spec,
                extras_spec=extra_specs,
                next_extras_spec=next_extra_specs,
            )

        builder.store.adder_signature_fn = adder_sig_fn


@dataclass
class ParallelSequenceAdderConfig:
    sequence_length: int = 20
    period: int = 10
    # use_next_extras: bool = False


class ParallelSequenceAdder(Component):
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
        builder.store.sequence_length = self.config.sequence_length

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        assert not hasattr(builder.store, "adder_priority_fn")

        # Create custom priority functions for the adder
        priority_fns = {
            table_key: lambda x: 1.0
            for table_key in builder.store.table_network_config.keys()
        }

        adder = reverb_adders.ParallelSequenceAdder(
            priority_fns=priority_fns,
            client=builder.store.data_server_client,
            net_ids_to_keys=builder.store.unique_net_keys,
            sequence_length=self.config.sequence_length,
            table_network_config=builder.store.table_network_config,
            period=self.config.period,
            # use_next_extras=self.config.use_next_extras,
        )

        builder.store.adder = adder

    @staticmethod
    def config_class() -> Callable:
        """Returns the class of the config for this component."""
        return ParallelSequenceAdderConfig

    @staticmethod
    def name() -> str:
        """Returns the name of the component."""
        return "parallel_sequence_adder"


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

        builder.store.adder_signature_fn = adder_sig_fn
