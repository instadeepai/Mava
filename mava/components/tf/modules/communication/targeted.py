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

# TODO (Kevin): complete class for targeted communication

"""Broadcasted communication for multi-agent RL systems"""

from typing import Dict

import sonnet as snt

from mava.components.tf.architectures import BaseArchitecture
from mava.components.tf.modules.communication import BaseCommunicationModule


class TargetedCommunication(BaseCommunicationModule):
    """Multi-agent broadcasted communication architecture."""

    def __init__(self, architecture: BaseArchitecture) -> None:
        self._architecture = architecture

    def process_messages(
        self,
        messages: Dict[str, snt.Module],
    ) -> Dict[str, snt.Module]:
        """Perform some communication logic"""
