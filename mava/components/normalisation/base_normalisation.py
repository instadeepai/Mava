from abc import ABC, abstractmethod

from mava.components import Component
from mava.core_jax import SystemBuilder, SystemParameterServer


class BaseNormalisation(Component, ABC):
    @abstractmethod
    def on_building_init(self, builder: SystemBuilder) -> None:
        """Initialise normalisation parameters"""

    @abstractmethod
    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Initialise normalisation parameters in the store"""

    @abstractmethod
    def on_parameter_server_init(self, server: SystemParameterServer) -> None:
        """Stores the normalisation parameters on the parameter server"""
