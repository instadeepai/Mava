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

"""MARL system communication modules."""

from mava.components.tf.modules.communication.base import BaseCommunicationModule
from mava.components.tf.modules.communication.broadcasted import (
    BroadcastedCommunication,
)
from mava.components.tf.modules.communication.targeted import TargetedCommunication
