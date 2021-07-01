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

from typing import Dict

import sonnet as snt

from mava.components.tf.architectures import BaseArchitecture
from mava.components.tf.modules.mixing.base import BaseMixingModule
from mava.components.tf.networks.additive import AdditiveMixingNetwork


class AdditiveMixing(BaseMixingModule):
    """Multi-agent monotonic mixing architecture."""

    def __init__(self, architecture: BaseArchitecture) -> None:
        """Initializes the mixer."""
        super(AdditiveMixing, self).__init__()

        self._architecture = architecture
        self._agent_networks = self._architecture.create_actor_variables()

    def _create_mixing_layer(self, name: str = "mixing") -> snt.Module:
        # Instantiate additive mixing network
        self._mixed_network = AdditiveMixingNetwork(name)
        return self._mixed_network

    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        self._agent_networks["mixing"] = self._create_mixing_layer()
        self._agent_networks["target_mixing"] = self._create_mixing_layer()

        return self._agent_networks
