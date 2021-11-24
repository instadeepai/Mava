from typing import Any, List, Union

import launchpad as lp

from mava.utils import lp_utils


class NodeType:
    reverb = lp.ReverbNode
    corrier = lp.CourierNode


class launcher:
    """
    # This mava launcher can be used to easily switch between single or distributed computation.

    # Example code:
    # program = launcher(single_process=False, nodes_on_gpu=["trainer"], name="Football")

    # replay = program.add(builder.make_replay_tables, node_type=NodeType.reverb, name="replay")

    # trainer = program.add(builder.make_trainer, replay, node_type=NodeType.corrier, name="trainer")

    # executors = [program.add(builder.make_executor, [replay, trainer],
    #             node_type=NodeType.corrier, name="executor") for _ in range(num_workers)]

    # variable_server = program.add(builder.make_coordinator, [trainer, executors],
    #             node_type=NodeType.corrier, name="coordinator")

    # program.launch()
    """

    def __init__(
        self, single_process: bool, nodes_on_gpu: List = [], name: str = "System"
    ) -> None:
        self._single_process = single_process
        self._name = name
        if not single_process:
            self._program = lp.Program(name=name)
            self._nodes_on_gpu = nodes_on_gpu

    def add(
        self,
        node_fn: Any,
        arguments: Any = [],
        node_type: Union[lp.ReverbNode, lp.CourierNode] = NodeType.corrier,
        name: str = "Node",
    ) -> Any:
        # Create a list of arguments
        if type(arguments) is not list:
            arguments = [arguments]

        if self._single_process:
            raise NotImplementedError("Single process launching not implemented yet.")
        else:
            with self._program.group(name):
                node = self._program.add_node(node_type(node_fn, *arguments))
            return node

    def launch(self) -> None:
        if self._single_process:
            raise NotImplementedError("Single process launching not implemented yet.")
        else:
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
