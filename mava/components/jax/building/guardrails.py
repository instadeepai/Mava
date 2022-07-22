from dataclasses import dataclass
from typing import Callable, List, Optional, Type

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.core_jax import SystemBuilder


@dataclass
class ComponentDependencyGuardrailsConfig:
    pass


class ComponentDependencyGuardrails(Component):
    def __init__(
        self,
        config: ComponentDependencyGuardrailsConfig,
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
    def config_class() -> Optional[Callable]:
        """Optional class which specifies the dataclass/config object for the component.

        Returns:
            config class/dataclass for component.
        """
        return ComponentDependencyGuardrailsConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []
