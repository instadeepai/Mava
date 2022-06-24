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
        components.extend(list(set(additional_required_components)))

        # Create edge map
        edges_from = []
        for component in components:
            for required_component in component.required_components():
                component_str = self._component_to_str(component)
                required_component_str = self._component_to_str(required_component)
                print(component_str, "---> requires --->", required_component_str)
                edges_from.append((component_str, required_component_str))

        # Create graph
        graph = nx.DiGraph()
        graph.add_edges_from(edges_from)

        # Draw graph
        pos = nx.planar_layout(graph, scale=2)
        nx.draw(graph, pos=pos)
        nx.draw_networkx_labels(graph, pos=pos, font_size=8)
        plt.axis("off")
        plt.savefig("./component_dependency_map.png")

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
