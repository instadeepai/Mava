from typing import Dict, List, TypeAlias, Any

_ConfSingleValue: TypeAlias = bool | int | float
ConfigValue: TypeAlias = _ConfSingleValue | List[_ConfSingleValue] | Dict[str, _ConfSingleValue]


def find_replace(d: Dict[str, Any], replacements: Dict[str, ConfigValue]) -> Dict[str, ConfigValue]:
    """Recursively searches through a dictionary and replaces values for specified keys.

    Args:
        d: Dictionary to search through
        replacements: The keys and values to replace
    """

    def _find_replace_recursive(current_dict: Dict[str, ConfigValue]) -> Dict[str, ConfigValue]:
        """Helper function that recursively searches and replaces values."""
        for k, v in current_dict.items():
            if isinstance(v, dict):
                current_dict[k] = _find_replace_recursive(v)
            elif k in replacements:
                current_dict[k] = replacements[k]

        return current_dict

    return _find_replace_recursive(d)
