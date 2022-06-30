from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks import Callback
from mava.systems.jax import Builder
from tests.jax.hook_order_tracking import HookOrderTracking


class TestBuilder(HookOrderTracking, Builder):
    def __init__(
        self,
        components: List[Callback],
        global_config: SimpleNamespace,
    ) -> None:
        """Initialise the builder."""
        self.reset_hook_list()

        super().__init__(components=components, global_config=global_config)


@pytest.fixture
def test_builder() -> Builder:
    """Dummy builder with no components"""
    return TestBuilder(
        components=[],
        global_config=SimpleNamespace(),
    )
