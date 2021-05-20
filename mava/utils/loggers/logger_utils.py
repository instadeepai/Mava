from typing import Any

from mava.utils.loggers.base import Logger


def make_logger(label: str, **kwargs: Any) -> Logger:
    return Logger(label=label, **kwargs)
