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


"""System Trainer implementation."""

from typing import Any, Dict, Iterator, List, Optional
from types import SimpleNamespace
import reverb

from mava import types
from mava.callbacks import Callback, TrainerHookMixin
from mava.core_jax import SystemTrainer


class Trainer(SystemTrainer, TrainerHookMixin):
    """MARL trainer"""

    def __init__(
        self,
        attr: SimpleNamespace,
        components: List[Callback] = [],
    ):
        """_summary_

        Args:
            attr : _description_
            components : _description_.
        """
        self.attr = attr
        self.callbacks = components

        self.on_training_init_start()

        self.on_training_init()

        self.on_training_init_end()

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass"""
        self._inputs = inputs

        self.on_training_forward_start()

        self.on_training_forward()

        self.on_training_forward_end()

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""
        self.on_training_backward_start()

        self.on_training_backward()

        self.on_training_backward_end()

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Trainer forward and backward passes.

        Returns:
            losses
        """
        self.on_training_compute_step_start()

        self.on_training_compute_step()

        self.on_training_compute_step_end()

        return self.attr.losses

    def step(self) -> None:
        """Trainer forward and backward passes."""
        self.on_training_step_start()

        self.on_training_step()

        self.on_training_step_end()
