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


from typing import Any, Dict, Mapping, Sequence, Union

import launchpad as lp
import sonnet as snt
import tensorflow as tf
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.utils import lp_utils

from mava import specs as mava_specs
from mava.components.tf.networks import NetworkWithMaskedEpsilonGreedy
from mava.systems.tf import madqn

from . import helpers

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_type",
    "atari",
    "Pettingzoo environment type, e.g. atari (str).",
)
flags.DEFINE_string(
    "env_name",
    "maze_craze_v2",
    "Pettingzoo environment name, e.g. pong (str).",
)


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    epsilon: tf.Variable,
    q_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256),
    shared_weights: bool = False,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(q_networks_layer_sizes, Sequence):
        q_networks_layer_sizes = {key: q_networks_layer_sizes for key in specs.keys()}

    observation_networks = {}
    q_networks = {}
    behavior_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the shared observation network
        observation_network = networks.ResNetTorso()

        # Create the policy network.
        q_network = snt.Sequential(
            [
                networks.LayerNormMLP(q_networks_layer_sizes[key], activate_final=True),
                networks.NearZeroInitializedLinear(num_dimensions),
            ]
        )

        behavior_network = NetworkWithMaskedEpsilonGreedy(q_network, epsilon=epsilon)

        observation_networks[key] = observation_network
        q_networks[key] = q_network
        behavior_networks[key] = behavior_network

    return {
        "q_networks": q_networks,
        "observations": observation_networks,
        "behaviors": behavior_networks,
    }


def main(_: Any) -> None:
    environment_factory = lp_utils.partial_kwargs(
        helpers.make_environment, env_type=FLAGS.env_type, env_class=FLAGS.env_name
    )

    program = madqn.DistributedMADQN(
        environment_factory=environment_factory,
        network_factory=lp_utils.partial_kwargs(make_networks),
        num_executors=2,
    ).build()

    lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == "__main__":
    app.run(main)
