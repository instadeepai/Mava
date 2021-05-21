# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

# TODO (StJohn): complete class for transformed mixing

"""Mixing for multi-agent RL systems"""

from typing import Dict

import sonnet as snt

from mava.components.tf.architectures import BaseArchitecture
from mava.components.tf.modules.mixing import BaseMixingModule


class QTranBase(BaseMixingModule):
    """Multi-agent mixing architecture."""

    def __init__(
        self,
        architecture: BaseArchitecture,
    ) -> None:
        """Initializes the mixer.
        Args:
            architecture: the BaseArchitecture used.
        """
        super(QTranBase, self).__init__()

        self._architecture = architecture

    def create_mixing_layer(self) -> Dict[str, Dict[str, snt.Module]]:
        # Implement method from base class
        # TODO Call mixing network from networks folder.
        return {}

    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        # Implement method from base class
        return {}


class QTranAlt(BaseMixingModule):
    """Multi-agent mixing architecture."""

    def __init__(
        self,
        architecture: BaseArchitecture,
    ) -> None:
        """Initializes the mixer.
        Args:
            architecture: the BaseArchitecture used.
        """
        super(QTranAlt, self).__init__()

        self._architecture = architecture

    def create_mixing_layer(self) -> Dict[str, Dict[str, snt.Module]]:
        # Implement method from base class
        # TODO Call mixing network from networks folder.
        return {}

    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        # Implement method from base class
        return {}
