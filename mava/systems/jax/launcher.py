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
import copy
from typing import Any, Dict, List, Union

import launchpad as lp
import reverb

from mava.core_jax import SystemBuilder
from mava.utils import lp_utils


class NodeType:
    reverb = lp.ReverbNode
    courier = lp.CourierNode


def copy_store(builder: SystemBuilder, multi_process: bool) -> SystemBuilder:
    if multi_process:
        # Note: It is unnecessary to copy when Launchpad is used.
        return builder
    else:
        # Note: Add store.program to the memo to avoid copying
        # it.
        memo = {}
        memo[id(builder.store.program)] = builder.store.program
        buildercopy = copy.deepcopy(builder, memo=memo)

        return buildercopy


class Launcher:
    """This mava launcher can be used to launch multi-node systems using either single \
        or distributed computation."""

    def __init__(
        self,
        multi_process: bool,
        nodes_on_gpu: List = [],
        sp_trainer_period: int = 10,
        sp_evaluator_period: int = 10,
        name: str = "System",
        terminal: str = "current_terminal",
        lp_launch_type: Union[
            str, lp.LaunchType
        ] = lp.LaunchType.LOCAL_MULTI_PROCESSING,
    ) -> None:
        """_summary_
        Args:
            multi_process : _description_.
            nodes_on_gpu : _description_.
            sp_trainer_period : _description_.
            sp_evaluator_period : _description_.
            name : _description_.
            terminal : terminal for launchpad processes to be shown on
            lp_launch_type: launchpad launch type to be used by system
        """
        self._multi_process = multi_process
        self._name = name
        self._sp_trainer_period = sp_trainer_period
        self._sp_evaluator_period = sp_evaluator_period
        self._terminal = terminal
        self._lp_launch_type = lp_launch_type
        if multi_process:
            self._program = lp.Program(name=name)
            self._nodes_on_gpu = nodes_on_gpu
        else:
            self._nodes: List = []
            self._node_dict: Dict = {
                "data_server": None,
                "parameter_server": None,
                "executor": None,
                "evaluator": None,
                "trainer": None,
            }

    def add(
        self,
        node_fn: Any,
        arguments: Any = [],
        node_type: Union[lp.ReverbNode, lp.CourierNode] = NodeType.courier,
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
            if name not in self._node_dict:
                raise ValueError(
                    f"{name} is not a valid node name."
                    + "Single process currently only supports "
                    + "nodes named: {list(self._node_dict.keys())}"
                )
            elif self._node_dict[name] is not None:
                raise ValueError(
                    f"Node named {name} initialised more than onces."
                    + "Single process currently only supports one node per type."
                )

            process = node_fn(*arguments)
            if node_type == lp.ReverbNode:
                # Assigning server to self to keep it alive.
                self._replay_server = reverb.Server(process, port=None)
                process = reverb.Client(f"localhost:{self._replay_server.port}")
            self._nodes.append(process)
            self._node_dict[name] = process
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
                launch_type=self._lp_launch_type,
                terminal=self._terminal,
                local_resources=local_resources,
            )
        else:
            episode = 1
            executor_steps = 0

            _ = self._node_dict["data_server"]
            _ = self._node_dict["parameter_server"]
            executor = self._node_dict["executor"]
            evaluator = self._node_dict["evaluator"]
            trainer = self._node_dict["trainer"]

            while True:
                executor_stats = executor.run_episode_and_log()

                if episode % self._sp_trainer_period == 0:
                    _ = trainer.step()  # logging done in trainer
                    print("Performed trainer step.")
                if episode % self._sp_evaluator_period == 0:
                    _ = evaluator.run_episode_and_log()
                    print("Performed evaluator run.")

                print(f"Episode {episode} completed.")
                episode += 1
                executor_steps += executor_stats["episode_length"]
