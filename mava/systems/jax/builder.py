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

"""Jax-based Mava system builder implementation."""

from typing import Any, List


class Builder:
    def __init__(
        self,
        components: List[Any],
    ) -> None:
        """System building init

        Args:
            components: system callback components
        """

        self.callbacks = components

    def build(self) -> None:
        """Build the system."""
        pass

    def launch(self) -> None:
        """Launch the system"""
        pass
