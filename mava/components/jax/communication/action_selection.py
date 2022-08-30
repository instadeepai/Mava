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

"""Components for GDN communication during action selection."""

from dataclasses import dataclass
from typing import List, Type

from mava.callbacks import Callback
from mava.components.jax import Component


@dataclass
class FeedForwardGdnCommunicationConfig:
    pass


class FeedForwardGdnCommunication(Component):
    def __init__(
        self,
        config: FeedForwardGdnCommunicationConfig = FeedForwardGdnCommunicationConfig(),
    ):
        """Component builds a GDN communication graph from the environment."""
        self.config = config

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "feed_forward_gdn_communication"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []
