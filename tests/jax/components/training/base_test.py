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

from dataclasses import dataclass

import pytest

from mava.components.jax.training.base import Loss, Step
from mava.core_jax import SystemTrainer


@dataclass
class MockLossConfig:
    clipping_epsilon: float = 0.2
    clip_value: bool = True
    entropy_cost: float = 0.01
    value_cost: float = 0.5


class MockLoss(Loss):
    """Creates mock loss class for trainer_base"""

    def __init__(self, config: MockLossConfig = MockLossConfig()) -> None:
        """Dummy function"""
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """Dummy function"""
        pass


class MockStep(Step):
    """Creates mock step class for trainer_base"""

    def __init__(arg) -> None:
        """Dummy function"""
        pass

    def on_training_init_start(self, trainer: SystemTrainer) -> None:
        """Dummy function"""
        pass

    def on_training_step_fn(self, trainer: SystemTrainer) -> None:
        """Dummy function"""
        pass

    def step(self, trainer: SystemTrainer) -> None:
        """Dummy function"""
        pass


@pytest.fixture
def base_loss() -> MockLoss:
    """Creates mock loss fixture"""

    test_loss = MockLoss(config=MockLossConfig())
    return test_loss


@pytest.fixture
def base_step() -> MockStep:
    """Creates mock step fixture"""

    test_step = MockStep()
    return test_step


def test_loss(base_loss: Loss) -> None:
    """Tests if loss naming is correct"""

    name = base_loss.name()
    assert name == "loss"


def test_step(base_step: Step) -> None:
    """Tests if step naming is correct"""

    name = base_step.name()
    assert name == "sgd_step"
