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


"""MADDPG trainer implementation."""

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import losses
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import mava
from mava import types as mava_types
from mava.adders.reverb.base import Trajectory
from mava.components.tf.losses.sequence import recurrent_n_step_critic_loss
from mava.systems.tf.variable_utils import VariableClient
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()

from mava.core import SystemTrainer
from mava.callbacks import Callback
from mava.systems.callback_hook import SystemCallbackHookMixin


class Trainer(SystemTrainer, SystemCallbackHookMixin):
    """MARL trainer"""

    def __init__(
        self,
        components: List[Callback] = [],
    ):
        """[summary]

        Args:
            components (List[Callback], optional): [description]. Defaults to [].
        """
        self.callbacks = components

        self.on_training_init_start(self)

        self.on_training_init(self)

        self.on_training_init_end(self)

    def _update_target_networks(self) -> None:
        """Sync the target network parameters with the latest online network
        parameters"""

    def _transform_observations(
        self, obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Transform the observatations using the observation networks of each agent."""

    def _get_feed(
        self,
        transitions: Dict[str, Dict[str, np.ndarray]],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """get data to feed to the agent networks"""

    def _step(self) -> Dict:
        """Trainer forward and backward passes."""

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass"""

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

    def step(self) -> None:
        """trainer step to update the parameters of the agents in the system"""
