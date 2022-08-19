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
from typing import Any, Dict, List, Optional, Union

import launchpad as lp
import reverb

from mava.utils import lp_utils
from mava.utils.builder_utils import copy_node_fn


class NodeType:
    reverb = lp.ReverbNode
    courier = lp.CourierNode


class Launcher:
    """This mava launcher can be used to launch multi-node systems using either single \
        or distributed computation."""

    def __init__(
        self,
        multi_process: bool,
        nodes_on_gpu: List = [],
        single_process_trainer_period: int = 1,
        single_process_evaluator_period: int = 10,
        single_process_max_episodes: Optional[int] = None,
        name: str = "System",
        terminal: str = "current_terminal",
        lp_launch_type: Union[
            str, lp.LaunchType
        ] = lp.LaunchType.LOCAL_MULTI_PROCESSING,
    ) -> None:
        """_summary_

        Args:
            multi_process : whether to use launchpad to run nodes on separate processes.
            nodes_on_gpu : which nodes should be run on the GPU.
            single_process_trainer_period : number of episodes between single process
                trainer steps.
            single_process_evaluator_period : num episodes between single process
                evaluator steps.
            single_process_max_episodes: maximum number of episodes to run
                before termination.
            name : launchpad program name.
            terminal : terminal for launchpad processes to be shown on.
            lp_launch_type: launchpad launch type.
        """
        self._multi_process = multi_process
        self._name = name
        self._single_process_trainer_period = single_process_trainer_period
        self._single_process_evaluator_period = single_process_evaluator_period
        self._single_process_max_episodes = single_process_max_episodes
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
                node_fn = copy_node_fn(node_fn)
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

            node_fn = copy_node_fn(node_fn)
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
            step = 1
            executor_steps = 0

            data_server = self._node_dict["data_server"]
            _ = self._node_dict["parameter_server"]
            executor = self._node_dict["executor"]
            evaluator = self._node_dict["evaluator"]
            trainer = self._node_dict["trainer"]

            # getting the maximum queue size
            queue_threshold = data_server.server_info()["trainer"].max_size

            while (
                self._single_process_max_episodes is None
                or episode <= self._single_process_max_episodes
            ):
                # if the queue is too full we skip the executor to ensure that the
                # executor won't hang when trying to push experience
                if data_server.server_info()["trainer"].current_size < int(
                    queue_threshold * 0.75
                ):
                    executor_stats = executor.run_episode_and_log()
                    executor_steps += executor_stats["episode_length"]

                    print(f"Episode {episode} completed.")
                    episode += 1

                # if the queue has less than sample_batch_size samples in it we skip
                # the trainer to ensure that the trainer won't hang
                if (
                    data_server.server_info()["trainer"].current_size
                    >= trainer.store.global_config.sample_batch_size
                    and step % self._single_process_trainer_period == 0
                ):
                    _ = trainer.step()  # logging done in trainer
                    print("Performed trainer step.")
                if step % self._single_process_evaluator_period == 0:
                    _ = evaluator.run_episode_and_log()
                    print("Performed evaluator run.")

                step += 1
