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

import logging
from types import SimpleNamespace
from typing import List

from mava.callbacks import Callback, TrainerHookMixin
from mava.core_jax import SystemTrainer
from mava.utils.jax_training_utils import set_growing_gpu_memory_jax
from mava.utils.training_utils import set_growing_gpu_memory

set_growing_gpu_memory_jax()
set_growing_gpu_memory()


class Trainer(SystemTrainer, TrainerHookMixin):
    """MARL trainer"""

    def __init__(
        self,
        store: SimpleNamespace,
        components: List[Callback] = [],
    ):
        """Initialise the trainer.

        Args:
            store: builder store.
            components: components in the system.
        """
        self.store = store
        self.callbacks = components

        self.on_training_init_start()

        self.on_training_utility_fns()

        self.on_training_loss_fns()

        self.on_training_step_fn()

        self.on_training_init()

        self.on_training_init_end()

    def step(self) -> None:
        """Trainer forward and backward passes."""
        self.on_training_step_start()

        self.on_training_step()

        self.on_training_step_end()

    def run(self) -> None:
        """Step the trainer in an infinite loop."""

        # Run the trainer.
        while True:
            try:
                self.step()
            except Exception as e:
                logging.exception(f"{e} the trainer failed")
                self.store.trainer_parameter_client.set_and_wait(
                    {"evaluator_or_trainer_failed": True}
                )
                break
