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


"""Stabilising for multi-agent RL systems"""
from typing import Dict, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from tensorflow.python.ops.gen_array_ops import concat

from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules.stabilising import BaseStabilisationModule


class FingerPrintStabalisation(BaseStabilisationModule):
    """Multi-agent stabalisation architecture."""

    def __init__(
        self,
    ) -> None:
        self._spec = tf.ones((2,), dtype="float32")
        self._scale_factor = 100000

    def get_spec(self):
        return self._spec

    def apply_to_architecture(self, architecture: DecentralisedValueActor):
        for key, spec in architecture._actor_specs.items():
            spec = tf.ones(shape=spec.shape, dtype="float32")
            spec = tf.concat([spec, self._spec], axis=0)
            architecture._actor_specs[key] = spec

    def trainer_hook(
        self,
        o_tm1_trans: tf.Tensor,
        o_t_trans: tf.Tensor,
        e_tm1: Dict[str, np.array],
        e_t: Dict[str, np.array],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Get fingerprints from extras.
        f_tm1 = tf.convert_to_tensor(e_tm1["fingerprint"], dtype="float32")
        f_t = tf.convert_to_tensor(e_t["fingerprint"], dtype="float32")

        # Concatenate fingerprints to observation embeddings.
        o_tm1_fprint = tf.concat([o_tm1_trans, f_tm1], axis=1)
        o_t_fprint = tf.concat([o_t_trans, f_t], axis=1)

        return o_tm1_fprint, o_t_fprint

    def executor_act_hook(
        self, observation: tf.Tensor, info: Dict[str, tf.Tensor]
    ) -> np.array:
        fingerprint = [info["epsilon"], info["trainer_step"] / self._scale_factor]
        fingerprint = np.array(fingerprint)
        fingerprint = tf.convert_to_tensor(fingerprint, dtype="float32")

        observation = tf.concat([observation, fingerprint], axis=0)
        return observation

    def executor_observe_hook(
        self, extras: Dict[str, np.array], info: Dict[str, np.array]
    ) -> Dict[str, np.array]:
        # Compute fingerprint using info
        fingerprint = np.array(
            [info["epsilon"], info["trainer_step"] / self._scale_factor],
            dtype="float32",
        )

        # Add fingerprint to extras.
        extras["fingerprint"] = fingerprint

        return extras
