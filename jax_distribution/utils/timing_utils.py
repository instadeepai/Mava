import timeit
from typing import Any


class TimeIt:
    """Context manager for timing execution."""

    def __init__(self, tag: str, environment_steps: int = None) -> None:
        """Initialise the context manager."""
        self.tag = tag
        self.environment_steps = environment_steps

    def __enter__(self) -> "TimeIt":
        """Start the timer."""
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args: Any) -> None:
        """Print the elapsed time."""
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.environment_steps:
            msg += ", SPS=%.2e" % (self.environment_steps / self.elapsed_secs)
        print(msg)
