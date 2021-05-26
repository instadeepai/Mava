# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

from typing import Any

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class ActorNetwork(snt.Module):
    def __init__(
        self,
        n_hidden_unit1: int,
        n_hidden_unit2: int,
        n_hidden_unit3: int,
        n_actions: int,
        logprob_epsilon: float,
        observation_netork: snt.Module,
    ):
        super(ActorNetwork, self).__init__()
        self.logprob_epsilon = tf.Variable(logprob_epsilon)
        self.observation_network = observation_netork
        w_bound = tf.Variable(3e-3)
        self.hidden1 = snt.Linear(n_hidden_unit1)
        self.hidden2 = snt.Linear(n_hidden_unit2)
        self.hidden3 = snt.Linear(n_hidden_unit3)

        self.mean = snt.Linear(
            n_actions,
            w_init=snt.initializers.RandomUniform(-w_bound, w_bound),
            b_init=snt.initializers.RandomUniform(-w_bound, w_bound),
        )
        self.log_std = snt.Linear(
            n_actions,
            w_init=snt.initializers.RandomUniform(-w_bound, w_bound),
            b_init=snt.initializers.RandomUniform(-w_bound, w_bound),
        )

    def __call__(self, x: Any) -> Any:
        """forward call for sonnet module"""
        x = self.observation_network(x)
        x = self.hidden1(x)
        x = tf.nn.relu(x)
        x = self.hidden2(x)
        x = tf.nn.relu(x)
        x = self.hidden3(x)
        x = tf.nn.relu(x)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - tf.math.log(
            1.0 - tf.pow(squashed_actions, 2) + self.logprob_epsilon
        )
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)
        return squashed_actions, logprob
