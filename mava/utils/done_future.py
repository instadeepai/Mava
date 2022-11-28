from concurrent import futures
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")  # type of result


class DoneFuture(futures.Future, Generic[T]):
    """A fake future that is always done. Duck-type to behave like a future"""

    def __init__(self, result: T):
        """Creates a future that is always done"""
        self._result = result

    def result(self, timeout: Optional[float] = None) -> T:
        """Returns the result of a future"""
        return self._result

    def add_done_callback(self, fn: Callable[..., Any]) -> None:
        """Calls fn(self) immediately as DoneFuture is always done"""
        fn(self)
        return None

    def done(self) -> bool:
        """Returns True as a DoneFuture is always done"""
        return True

    def running(self) -> bool:
        """Returns false as the future is complete and cannot be running"""
        return False

    def cancelled(self) -> bool:
        """Returns false as the future is complete and cannot be cancelled"""
        return False

    def cancel(self) -> bool:
        """Returns false as the future is complete and cannot be cancelled"""
        return True
