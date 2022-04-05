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

"""General launcher for systems"""
from typing import Any, List, Union

import launchpad as lp
import reverb

from mava.utils import lp_utils


class NodeType:
    reverb = lp.ReverbNode
    corrier = lp.CourierNode


class Launcher:
    """This mava launcher can be used to launch multi-node systems using either single \
        or distributed computation."""

    def __init__(
        self,
        multi_process: bool,
        nodes_on_gpu: List = [],
        name: str = "System",
    ) -> None:
        """_summary_

        Args:
            multi_process : _description_
            nodes_on_gpu : _description_.
            name : _description_.
        """
        self._multi_process = multi_process
        self._name = name
        if multi_process:
            self._program = lp.Program(name=name)
            self._nodes_on_gpu = nodes_on_gpu
        else:
            self._nodes: List = []

    def add(
        self,
        node_fn: Any,
        arguments: Any = [],
        node_type: Union[lp.ReverbNode, lp.CourierNode] = NodeType.corrier,
        name: str = "Node",
    ) -> Any:
        """_summary_

        Args:
            node_fn : _description_
            arguments : _description_.
            node_type : _description_.
            name : _description_.

        Raises:
            NotImplementedError: _description_
        Returns:
            _description_
        """
        # Create a list of arguments
        if type(arguments) is not list:
            arguments = [arguments]

        if self._multi_process:
            with self._program.group(name):
                node = self._program.add_node(node_type(node_fn, *arguments))
            return node
        else:
            process = node_fn(*arguments)
            if node_type == lp.ReverbNode:

                # Assigning server to self to keep it alive.
                self._replay_server = reverb.Server(process, port=None)
                process = reverb.Client(f"localhost:{self._replay_server.port}")
            self._nodes.append(process)
            return process

    def get_nodes(self) -> List[Any]:
        """TODO: Add description here."""
        if self._multi_process:
            raise ValueError("Get nodes only implemented for single process setups.")

        return self._nodes

    def launch(self) -> None:
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        if self._multi_process:
            local_resources = lp_utils.to_device(
                program_nodes=self._program.groups.keys(),
                nodes_on_gpu=self._nodes_on_gpu,
            )

            lp.launch(
                self._program,
                lp.LaunchType.LOCAL_MULTI_PROCESSING,
                terminal="current_terminal",
                local_resources=local_resources,
            )
        else:
            raise NotImplementedError("Single process launching not implemented yet.")
