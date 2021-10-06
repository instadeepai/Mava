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
from mava.utils.sort_utils import sort_str_num

train_utils.set_growing_gpu_memory()

from mava.callbacks import Callback
from mava.systems.callback_hook import SystemCallbackHookMixin


class SystemTrainer(mava.Trainer, Callback):
    """[summary]

    Args:
        mava ([type]): [description]
        Callback ([type]): [description]

    Returns:
        [type]: [description]
    """


class OnlineSystemTrainer(SystemTrainer, SystemCallbackHookMixin):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

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

        self.on_training_init_end(self)

    # To be completed...
