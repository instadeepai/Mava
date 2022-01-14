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

"""Mava system implementation."""

from mava.systems.launcher import Launcher

from mava.callbacks import SystemCallbackHookMixin
from mava.core import SystemBuilder


class System(SystemCallbackHookMixin):
    """MARL system."""

    def __init__(self, builder: SystemBuilder, program: Launcher):
        """[summary]

        Args:
            builder (SystemBuilder): [description]
        """

        self.builder = builder
        self.program = program
        self.callbacks = builder.callbacks

    def build(self):
        self.builder.build(self.program)

    def launch(self):
        self.program.launch()