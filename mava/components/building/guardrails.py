from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Type

import matplotlib.pyplot as plt
import networkx as nx

from mava.callbacks import Callback
from mava.components import Component
from mava.core_jax import SystemBuilder


class ComponentDependencyGuardrails(Component):
    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ):
        """Save the config"""
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Check that all component requirements are satisfied"""
        for component in builder.callbacks:
            for required_component in component.required_components():
                required_component_present = False

                for system_component in builder.callbacks:
                    if isinstance(system_component, required_component):
                        required_component_present = True
                        break

                if not required_component_present:
                    raise Exception(
                        f"""
                        Component {component} requires other components
                        {component.required_components()}
                        to be present in the system.
                        Component {required_component} is not in the system.
                        """
                    )

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "component_dependency_guardrails"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []


@dataclass
class ComponentDependencyDebuggerConfig:
    show_full_component_path: bool = False
    fig_width_ratios = [1, 5]
    fig_inch_width: float = 18.5
    fig_inch_height: float = 10.5
    component_name_fontsize: int = 12
    node_label_font_size: int = 8
    required_node_color: str = "#ADD8E6"  # light blue
    leaf_required_node_color: str = "#90EE90"  # light green
    detached_node_color: str = "#F0F0F0"  # very light gray
    dependent_node_color: str = "#D3D3D3"  # light gray
    cycle_edge_color: str = "#FF0000"  # red
    default_edge_color: str = "#000000"  # black
    layout_scale: int = 2
    node_size_increment: int = 80
    min_node_size: int = 190
    dependency_graph_save_path: str = "./component_dependency_map.png"


class ComponentDependencyDebugger(Component):
    def __init__(
        self,
        config: ComponentDependencyDebuggerConfig,
    ):
        """Save the config"""
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:  # noqa: C901
        """Compute component dependencies and save graph."""
        components: List = builder.callbacks[:]

        # Add required components to list
        all_required_components = []
        for component in components:
            all_required_components.extend(component.required_components())
        components.extend(all_required_components)

        # Sort components alphabetically and remove duplicates
        unique_component_strings = []
        unique_components = []
        for component in components:
            component_str = self._component_to_str(component)
            if component_str not in unique_component_strings:
                unique_component_strings.append(component_str)
                unique_components.append(component)
        components = sorted(unique_components, key=lambda x: self._component_to_str(x))

        # Map from unique int IDs to components
        id_to_component = {}
        for i, component in enumerate(components):
            id_to_component[i] = self._component_to_str(component)
        component_to_id = {component: i for i, component in id_to_component.items()}

        # Create edge map
        edges = []
        for component in components:
            component_str = self._component_to_str(component)

            for required_component in component.required_components():
                required_component_str = self._component_to_str(required_component)

                print(component_str, "---> requires --->", required_component_str)
                edges.append(
                    (
                        component_to_id[component_str],
                        component_to_id[required_component_str],
                    )
                )
        to_nodes = [node_2 for (node_1, node_2) in edges]
        from_nodes = [node_1 for (node_1, node_2) in edges]

        # Create graph
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        graph.add_nodes_from(id_to_component.keys())

        # Create subplots
        plt.clf()
        fig, (left_ax, right_ax) = plt.subplots(
            nrows=1, ncols=2, gridspec_kw={"width_ratios": self.config.fig_width_ratios}
        )
        fig.set_size_inches(self.config.fig_inch_width, self.config.fig_inch_height)

        # Draw manual legend
        text = ""
        for i in range(len(components)):
            text += str(i) + " : " + id_to_component[i] + "\n"
        plt.text(
            x=0,
            y=0.5,
            s=text,
            horizontalalignment="left",
            verticalalignment="center",
            transform=left_ax.transAxes,
            fontsize=self.config.component_name_fontsize,
        )
        left_ax.axis("off")

        # Compute node colors
        node_color = []
        for node in graph:
            if node in to_nodes and node not in from_nodes:  # Leaf of graph
                node_color.append(self.config.leaf_required_node_color)
            elif node in to_nodes:  # Other components depend on it
                node_color.append(self.config.required_node_color)
            elif node in from_nodes:  # It depends on other nodes
                node_color.append(self.config.dependent_node_color)
            else:  # No adjacent edges
                node_color.append(self.config.detached_node_color)

        # Compute cycles
        cycles = sorted(nx.simple_cycles(graph))
        for cycle in cycles:
            print("Circular dependency found: ", end="")
            print([id_to_component[node] for node in cycle])
        cycle_edges = []
        for cycle in cycles:
            current_node = cycle[-1]
            for node in cycle:
                cycle_edges.append((current_node, node))
                current_node = node

        # Compute edge colors
        edge_color = []
        for edge in graph.edges:
            if edge in cycle_edges:  # Flag cycle edges
                edge_color.append(self.config.cycle_edge_color)
            else:
                edge_color.append(self.config.default_edge_color)

        # Draw graph
        pos = nx.circular_layout(graph, scale=self.config.layout_scale)
        degrees = nx.degree(graph)
        degrees = [
            (degrees[node] - 1) * self.config.node_size_increment
            + self.config.min_node_size
            for node in graph.nodes()
        ]
        nx.draw(
            graph,
            pos=pos,
            ax=right_ax,
            node_color=node_color,
            node_size=degrees,
            edge_color=edge_color,
        )
        nx.draw_networkx_labels(
            graph, pos=pos, font_size=self.config.node_label_font_size, ax=right_ax
        )
        right_ax.axis("off")

        # Save figure
        print("Saving figure to '", self.config.dependency_graph_save_path, "'", sep="")
        plt.savefig(
            self.config.dependency_graph_save_path,
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.5,
        )

    def _component_to_str(self, component: Callback) -> str:
        """Convert a component to a string representation."""
        component_str = str(component)[1:-1]  # Trim edge brackets
        component_str = component_str.replace("'", "")  # Replace quotations

        # Only the component path
        if component_str[:5] == "class":
            component_str = component_str.split(" ")[1]
        else:
            component_str = component_str.split(" ")[0]

        # Strip leading `mava.components.` if it exists
        if component_str.split(".")[:3] == ["mava", "components", "jax"]:
            component_str = ".".join(component_str.split(".")[3:])

        # Only show component name
        if not self.config.show_full_component_path:
            component_str = component_str.split(".")[-1]

        return component_str

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "component_dependency_debugger"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []
