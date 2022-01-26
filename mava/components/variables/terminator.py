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

"""Generic variable checkpointer component for systems"""

from typing import Dict

import launchpad as lp

from mava.core import SystemVariableServer
from mava.callbacks import Callback
from mava.utils.training_utils import check_count_condition


class SystemTerminator(Callback):
    """A terminator object to terminate system training and execution according to
    a pre-specified termintation condition."""

    def __init__(
        self,
        termination_condition: Dict = None,
    ) -> None:
        """[summary]

        Args:
            termination_condition (Dict, optional): [description]. Defaults to None.
        """
        self._termination_condition = termination_condition

        self._terminal_key, self._terminal_count = check_count_condition(
            self._termination_condition
        )

    def on_variables_run_server_loop_termination(
        self, server: SystemVariableServer
    ) -> None:
        if self._termination_condition is not None:
            current_count = float(server.variables[self._terminal_key])
            if current_count >= self._terminal_count:
                lp.stop()
