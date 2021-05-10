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

from typing import Optional, Sequence

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.specs import EnvironmentSpec
from acme.tf import networks


class DIALPolicy(snt.RNNCore):
    def __init__(
        self,
        action_spec: EnvironmentSpec,
        message_spec: EnvironmentSpec,
        gru_hidden_size: int,
        gru_layers: int,
        task_mlp_size: Sequence,
        message_in_mlp_size: Sequence,
        message_out_mlp_size: Sequence,
        output_mlp_size: Sequence,
        name: str = None,
    ):
        super(DIALPolicy, self).__init__(name=name)
        self._action_spec = action_spec
        self._action_dim = np.prod(self._action_spec.shape, dtype=int)
        self._message_spec = message_spec
        self._message_dim = np.prod(self._message_spec.shape, dtype=int)

        self._gru_hidden_size = gru_hidden_size

        self.task_mlp = networks.LayerNormMLP(
            task_mlp_size,
            activate_final=True,
        )

        self.message_in_mlp = networks.LayerNormMLP(
            message_in_mlp_size,
            activate_final=True,
        )

        self.message_in_mlp = networks.LayerNormMLP(
            message_out_mlp_size,
            activate_final=True,
        )

        self.gru = snt.GRU(gru_hidden_size)

        self.output_mlp = snt.Sequential(
            [
                networks.LayerNormMLP(output_mlp_size, activate_final=True),
                networks.NearZeroInitializedLinear(2),
                # networks.TanhToSpec(self._action_spec),
            ]
        )

        self.message_out_mlp = snt.Sequential(
            [
                networks.LayerNormMLP(message_out_mlp_size, activate_final=True),
                networks.NearZeroInitializedLinear(self._message_dim),
                # networks.TanhToSpec(self._message_spec),
            ]
        )

        self._gru_layers = gru_layers

    def initial_state(self, batch_size: int = 1) -> snt.Module:
        return self.gru.initial_state(batch_size)

    def initial_message(self, batch_size: int = 1) -> snt.Module:
        return tf.zeros([batch_size, self._message_dim], dtype=tf.float32)

    def __call__(
        self,
        x: snt.Module,
        state: Optional[snt.Module] = None,
        message: Optional[snt.Module] = None,
    ) -> snt.Module:
        if state is None:
            state = self.initial_state()
        if message is None:
            message = tf.zeros(self._message_dim, dtype=tf.float32)

        x_task = self.task_mlp(x)
        x_message = self.message_in_mlp(message)
        x = tf.concat([x_task, x_message], axis=1)

        x, state = self.gru(x, state)

        x_output = self.output_mlp(x)
        x_message = self.message_out_mlp(x)

        return (x_output, x_message), state
