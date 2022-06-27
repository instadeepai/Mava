from dataclasses import dataclass
from typing import Callable, List, Optional, Type

import matplotlib.pyplot as plt
import networkx as nx

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.core_jax import SystemBuilder


@dataclass
class ComponentDependenciesConfig:
    show_full_component_path: bool = False


class ComponentDependencyDebugger(Component):
    def __init__(
        self,
        config: ComponentDependenciesConfig,
    ):
        """Save the config"""
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Compute component dependencies and save graph."""
        components: List = builder.callbacks[:]

        # Add required components to list
        additional_required_components = []
        for component in components:
            additional_required_components.extend(component.required_components())
        components.extend(additional_required_components)

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
        edges_from = []
        for component in components:
            for required_component in component.required_components():
                component_str = self._component_to_str(component)
                required_component_str = self._component_to_str(required_component)
                print(component_str, "---> requires --->", required_component_str)
                edges_from.append(
                    (
                        component_to_id[component_str],
                        component_to_id[required_component_str],
                    )
                )
        to_nodes = [node_2 for (node_1, node_2) in edges_from]

        # Create graph
        graph = nx.DiGraph()
        graph.add_edges_from(edges_from)

        # Create subplots
        fig = plt.figure(1)
        plt.clf()
        fig, (left_ax, right_ax) = plt.subplots(
            nrows=1, ncols=2, gridspec_kw={"width_ratios": [1, 5]}
        )
        fig.set_size_inches(18.5, 10.5)

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
            fontsize=12,
        )
        left_ax.axis("off")

        # Compute node colors
        color_map = []
        for node in graph:
            if node in to_nodes:
                color_map.append("#FFCCCB")
            else:
                color_map.append("#D3D3D3")

        # Draw graph
        pos = nx.spring_layout(graph, scale=1, iterations=2)
        nx.draw(graph, pos=pos, ax=right_ax, node_color=color_map)
        nx.draw_networkx_labels(graph, pos=pos, font_size=8, ax=right_ax)
        right_ax.axis("off")

        # Save figure
        save_path = "./component_dependency_map.png"
        print("Saving figure to '", save_path, "'", sep="")
        plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.5)
        # raise Exception("TODO: remove")

    def _component_to_str(self, component: Callback) -> str:
        """Convert a component to a string representation."""
        component_str = str(component)[1:-1]  # Trim edge brackets
        component_str = component_str.replace("'", "")  # Replace quotations
        # Only the component path
        if component_str[:5] == "class":
            component_str = component_str.split(" ")[1]
        else:
            component_str = component_str.split(" ")[0]

        # Strip leading `mava.components.jax.` if it exists
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
    def config_class() -> Optional[Callable]:
        """Optional class which specifies the dataclass/config object for the component.

        Returns:
            config class/dataclass for component.
        """
        return ComponentDependenciesConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []
