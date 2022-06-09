# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from types import SimpleNamespace
from typing import Any

from mava.components.jax.training.advantage_estimation import GAE
from mava.core_jax import SystemTrainer


class MockTrainer(SystemTrainer):
    """Abstract system trainer."""

    def __init__(
        self,
    ) -> None:
        """System trainer init"""

        # Simple namespace for assigning system executor attributes dynamically
        self.store = SimpleNamespace(gae_fn=None)

        self._inputs: Any

    def step(self) -> None:
        """Trainer forward and backward passes."""
        pass


def test_gae_creation() -> None:
    """Test whether gae function is successfully created"""

    test_gae = GAE()
    mock_trainer = MockTrainer()
    test_gae.on_training_utility_fns(trainer=mock_trainer)

    assert mock_trainer.store.gae_fn is not None
