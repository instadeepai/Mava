import pytest

from mava.utils.done_future import DoneFuture

global_var = 1  # used to test the callback


@pytest.fixture
def done_future() -> DoneFuture:
    """Creates a DoneFuture"""
    return DoneFuture(42)


def test___init__(done_future: DoneFuture) -> None:
    """Tests that DoneFuture initialises properly"""
    assert done_future._result == 42


def test_result(done_future: DoneFuture) -> None:
    """Tests that DoneFuture always gets a result"""
    assert done_future.result() == 42


def test_add_done_callback(done_future: DoneFuture) -> None:
    """Tests that DoneFutures callbacks execute immediately"""

    def callback(fut: DoneFuture) -> None:
        global global_var
        global_var += fut.result()

    done_future.add_done_callback(callback)

    assert global_var == 43


def test_done(done_future: DoneFuture) -> None:
    """Tests that DoneFuture is always done"""
    assert done_future.done() == True  # noqa: E712


def test_running(done_future: DoneFuture) -> None:
    """Tests the DoneFuture is never running"""
    assert done_future.running() == False  # noqa: E712


def test_cancelled(done_future: DoneFuture) -> None:
    """Tests that DoneFuture is never cancelled"""
    assert done_future.cancelled() == False  # noqa: E712


def test_cancel(done_future: DoneFuture) -> None:
    """Tests that DoneFuture can never be cancelled"""
    assert done_future.cancel() == False  # noqa: E712
