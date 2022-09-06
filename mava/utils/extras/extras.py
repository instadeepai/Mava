import copy
from typing import Any, Dict, List


class UserDefinedExtrasFinder:
    """A class to find the values for the user-defined extras."""

    @staticmethod
    def find(store: Any, keys: List[str]) -> Dict:
        """Finds the information in the store.

        Args:
            store: the store of the executor.
            keys: the keys to look for in the store.

        Returns:
            a dictionary with (modified) keys and the values in store.
        """
        user_defined_extras = {}
        for key in keys:
            key_in_store = copy.deepcopy(key)
            modified_key = copy.deepcopy(key)
            if key == "network_keys":
                modified_key = "network_int_keys"
                key_in_store = "network_int_keys_extras"
            if key == "policy_info":
                key_in_store = "policy_info"
            value = store.__getattribute__(key_in_store)
            # value = store.extras_spec[key_in_store]
            user_defined_extras.update({modified_key: value})
        return user_defined_extras
