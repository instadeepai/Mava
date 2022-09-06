from typing import Any, Dict

from acme.core import Saveable as AcmeSaveable

"""SaveableWrapper as needed to use the Acme JAX checkpointer."""


class SaveableWrapper(AcmeSaveable):
    def __init__(self, state: Dict[str, Any]):
        """Initialise system state

        Args:
            state: a dictionary of variables to save
        """
        self.state = state

    def save(self) -> Dict[str, Any]:
        """Save system state

        Returns:
            system state.
        """
        return self.state

    def restore(self, state: Dict[str, Any]) -> None:
        """Restore system state

        Args:
            state: a dictionary of variables to save
        Returns:
            None.
        """
        self.state = state
