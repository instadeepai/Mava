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


class ParallelNStepTransitionAdder(Callback):
    def on_building_adder_signature(self, builder: BaseSystemBuilder):
        def adder_sig_fn(
            env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
        ) -> Any:
            return reverb_adders.ParallelNStepTransitionAdder.signature(
                env_spec, extra_specs
            )

        builder.adder_signature = adder_sig_fn


class ParallelSequenceAdder(Callback):
    def on_building_adder_signature(self, builder: BaseSystemBuilder):
        def adder_sig_fn(
            env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
        ) -> Any:
            return reverb_adders.ParallelSequenceAdder.signature(
                env_spec, self._config.sequence_length, extra_specs
            )

        builder.adder_signature = adder_sig_fn
