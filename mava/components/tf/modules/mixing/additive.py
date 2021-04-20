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
from mava.components.tf.modules.mixing import BaseMixingModule
from mava.components.tf.networks.additive import AdditiveMixingNetwork


class AdditiveMixing(BaseMixingModule):
    """Multi-agent monotonic mixing architecture."""

    def __init__(self, architecture: BaseArchitecture) -> None:
        """Initializes the mixer."""
        super(AdditiveMixing, self).__init__()

        self._architecture = architecture

    def _create_mixing_layer(self) -> snt.Module:
        # Instantiate additive mixing network
        return AdditiveMixingNetwork()

    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        # Implement method from base class
        networks = self._architecture.create_actor_variables()
        networks["mixing_network"] = self._create_mixing_layer()
        return networks
