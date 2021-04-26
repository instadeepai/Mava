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

"""Example running MADQN on the pettingzoo environment."""


from typing import Any, Mapping

import launchpad as lp
import tensorflow as tf
from absl import app, flags
from acme import types

from mava import specs as mava_specs
from mava.components.tf.networks import (
    MADQNAtariNetwork,
    NetworkWithMaskedEpsilonGreedy,
)
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.environments import pettingzoo_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "maze_craze_v2",
    "Pettingzoo environment name, e.g. pong (str).",
)


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    epsilon: tf.Variable = tf.Variable(0.05, trainable=False),
    shared_weights: bool = True,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    q_networks = {}
    policy_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the q-value network.
        q_network = MADQNAtariNetwork(num_dimensions)

        # Epsilon greedy policy network
        policy_network = NetworkWithMaskedEpsilonGreedy(q_network, epsilon=epsilon)

        q_networks[key] = q_network
        policy_networks[key] = policy_network
    return {
        "q_networks": q_networks,
        "policies": policy_networks,
    }


def main(_: Any) -> None:
    environment_factory = lp_utils.partial_kwargs(
        pettingzoo_utils.make_parallel_atari_environment,
        env_name=FLAGS.env_name,
    )

    program = madqn.DistributedMADQN(
        environment_factory=environment_factory,
        network_factory=lp_utils.partial_kwargs(make_networks),
        num_executors=2,
    ).build()

    lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == "__main__":
    app.run(main)
