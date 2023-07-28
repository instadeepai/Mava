import timeit
from typing import Any


class TimeIt:
    """Context manager for timing execution."""

    def __init__(self, tag: str, frames=None) -> None:
        """Initialise the context manager."""
        self.tag = tag
        self.frames = frames

    def __enter__(self) -> "TimeIt":
        """Start the timer."""
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args: Any) -> None:
        """Print the elapsed time."""
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.frames:
            msg += ", FPS=%.2e" % (self.frames / self.elapsed_secs)
        print(msg)
