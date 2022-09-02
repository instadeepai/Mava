from typing import Any, Dict

from acme.core import Saveable as AcmeSaveable

"""SaveableWrapper as needed to use the Acme JAX checkpointer."""


class SaveableWrapper(AcmeSaveable):
    def __init__(self, state: Dict[str, Any]):
        """Initialise system state

        Args:
            state (_type_):  system state represented by a dictionary of saved variables
        """
        self.state = state

    def save(self) -> Dict[str, Any]:
        """Save system state

        Returns:
            system state.
        """
        # TODO fix type
        return self.state

    def restore(self, state: Dict[str, Any]) -> None:
        """Restore system state

        Args:
            state (Any): system state represented by a dictionary of saved variables
        Returns:
            None.
        """
        self.state = state
