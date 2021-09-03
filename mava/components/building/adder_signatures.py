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

"""Commonly used adder signature components for system builders"""

from mava import specs
from mava.callbacks import Callback
from mava.systems.building import BaseSystemBuilder
from mava.adders import reverb as reverb_adders


class Adder(Callback):
    def on_building_adder_signature(self, builder: BaseSystemBuilder) -> None:
        """[summary]

        Args:
            builder (BaseSystemBuilder): [description]
        """
        pass


class ParallelNStepTransitionAdder(Adder):
    def on_building_adder_signature(self, builder: BaseSystemBuilder) -> None:
        def adder_sig_fn(
            env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
        ) -> Any:
            return reverb_adders.ParallelNStepTransitionAdder.signature(
                env_spec, extra_specs
            )

        builder.adder_signature_fn = adder_sig_fn


class ParallelSequenceAdder(Adder):
    def __init__(self, sequence_length: int = 20) -> None:
        """[summary]

        Args:
            sequence_length (int, optional): [description]. Defaults to 20.
        """
        self.sequence_length = sequence_length

    def on_building_adder_signature(self, builder: BaseSystemBuilder) -> None:
        def adder_sig_fn(
            env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
        ) -> Any:
            return reverb_adders.ParallelSequenceAdder.signature(
                env_spec, self.sequence_length, extra_specs
            )

        builder.adder_signature = adder_sig_fn
