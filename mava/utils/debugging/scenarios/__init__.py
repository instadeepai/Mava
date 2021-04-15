import importlib
from typing import Any


def load(name: str) -> Any:
    return importlib.import_module(f"mava.utils.debugging.scenarios.{name}")
