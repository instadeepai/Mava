# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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
    """Specify launchpad node types that systems can use."""

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
        is_test: Optional[bool] = False,
        wait: Optional[bool] = False,
    ) -> None:
        """Initialise the launcher.

        If multi-process, set up the launchpad program.
        Otherwise, create a dictionary for the nodes in the system.

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
            is_test : whether to set testing launchpad launch_type.
            wait: the worker manager will wait worker_manager.wait()
        """
        self._is_test = is_test
        self._wait = wait
        self._multi_process = multi_process
        self._name = name
        self._single_process_trainer_period = single_process_trainer_period
        self._single_process_evaluator_period = single_process_evaluator_period
        self._single_process_max_episodes = single_process_max_episodes
        self._terminal = terminal
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
        """Add a node to the system.

        If multi-processing, add a node to the existing launchpad program,
        grouped under the given name.
        This means that when multi-processing,
        you can have multiple nodes of the same name (e.g. executor).
        If system is single-process, only one node per name is allowed in the system.

        Args:
            node_fn : Function returning the system process that will run on the node.
            arguments : Arguments used when initialising the system process.
            node_type : Type of launchpad node to use.
            name : Node name (e.g. executor).

        Raises:
            ValueError: if single-process and node name is not supported.
            ValueError: if single-process and trying to init a node more than once.

        Returns:
            The system process or launchpad node.
        """
        # Create a list of arguments
        if type(arguments) is not list:
            arguments = [arguments]

        if self._multi_process:
            with self._program.group(name):
                # Save the current PID to manage the termination of the process
                if self._is_test:
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
                    f"Node named {name} initialised more than once."
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
        """Get the nodes of a single-process system.

        Raises:
            ValueError: if system is multi-process.

        Returns:
            System nodes.
        """
        if self._multi_process:
            raise ValueError("Get nodes only implemented for single process setups.")

        return self._nodes

    def launch(self) -> None:
        """Launch the launchpad program or start the single-process system loop."""
        if self._multi_process:
            if self._is_test:
                launch_type = lp.LaunchType.TEST_MULTI_THREADING
            else:
                launch_type = lp.LaunchType.LOCAL_MULTI_PROCESSING

            local_resources = lp_utils.to_device(
                program_nodes=self._program.groups.keys(),
                nodes_on_gpu=self._nodes_on_gpu,
            )

            worker_manager = lp.launch(
                self._program,
                launch_type=launch_type,
                terminal=self._terminal,
                local_resources=local_resources,
            )

            if self._wait:
                worker_manager.wait()

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
            queue_threshold = data_server.server_info()["trainer_0"].max_size

            while (
                self._single_process_max_episodes is None
                or episode <= self._single_process_max_episodes
            ):
                # if the queue is too full we skip the executor to ensure that the
                # executor won't hang when trying to push experience
                if data_server.server_info()["trainer_0"].current_size < int(
                    queue_threshold * 0.75
                ):
                    executor_stats = executor.run_episode_and_log()
                    executor_steps += executor_stats["episode_length"]

                    print(f"Episode {episode} completed.")
                    episode += 1

                # if the queue has less than sample_batch_size samples in it we skip
                # the trainer to ensure that the trainer won't hang
                if (
                    data_server.server_info()["trainer_0"].current_size
                    >= trainer.store.global_config.sample_batch_size
                    and step % self._single_process_trainer_period == 0
                ):
                    _ = trainer.step()  # logging done in trainer
                    print("Performed trainer step.")
                if step % self._single_process_evaluator_period == 0:
                    _ = evaluator.run_episode_and_log()
                    print("Performed evaluator run.")

                step += 1
