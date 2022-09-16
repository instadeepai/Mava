# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Unit tests for core Jax trainer component.

from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks import Callback
from mava.systems.trainer import Trainer
from tests.jax.hook_order_tracking import HookOrderTracking


class TestTrainer(HookOrderTracking, Trainer):
    __test__ = False

    def __init__(
        self,
        config: SimpleNamespace,
        components: List[Callback],
    ) -> None:
        """Initialise the trainer."""
        self.reset_hook_list()

        super().__init__(config, components)


@pytest.fixture
def mock_trainer() -> Trainer:
    """Create mock trainer."""

    trainer = TestTrainer(config=SimpleNamespace(), components=[])
    return trainer


def test_init_hook_order(mock_trainer: TestTrainer) -> None:
    """Test if init hooks are called in the correct order"""

    assert mock_trainer.hook_list == [
        "on_training_init_start",
        "on_training_utility_fns",
        "on_training_loss_fns",
        "on_training_step_fn",
        "on_training_init",
        "on_training_init_end",
    ]


def test_step_hook_order(mock_trainer: TestTrainer) -> None:
    """Test if step hooks are called in the correct order"""

    mock_trainer.reset_hook_list()
    mock_trainer.step()

    assert mock_trainer.hook_list == [
        "on_training_step_start",
        "on_training_step",
        "on_training_step_end",
    ]
