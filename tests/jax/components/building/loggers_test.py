import pytest

from mava.components.jax.building.loggers import Logger


class TestLogger(Logger):
    pass


@pytest.fixture
def test_logger() -> Logger:
    """Dummy system with zero components."""
    return TestLogger()


def test_assert_true(test_logger: Logger) -> None:
    assert True
