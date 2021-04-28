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

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import dm_env
import launchpad as lp
import sonnet as snt
import tensorflow as tf
import trfl
from absl import app, flags
from acme import types
from acme.tf import networks

from mava import specs as mava_specs
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.debugging.make_env import make_debugging_env
from mava.wrappers.debugging_envs import DebuggingEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Debugging environment name (str).",
)


def make_environment(
    evaluation: bool,
    env_name: str = "simple_spread",
    action_space: str = "discrete",
    num_agents: int = 3,
    render: bool = False,
) -> dm_env.Environment:

    assert action_space == "continuous" or action_space == "discrete"

    del evaluation

    """Creates a MPE environment."""
    env_module = make_debugging_env(env_name, action_space, num_agents)
    environment = DebuggingEnvWrapper(env_module, render=render)
    return environment


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    epsilon: tf.Variable = tf.Variable(1.0, trainable=False),
    q_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256),
    shared_weights: bool = True,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(q_networks_layer_sizes, Sequence):
        q_networks_layer_sizes = {key: q_networks_layer_sizes for key in specs.keys()}

    policy_networks = {}
    q_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the policy network.
        q_network = snt.Sequential(
            [
                networks.LayerNormMLP(q_networks_layer_sizes[key], activate_final=True),
                networks.NearZeroInitializedLinear(num_dimensions),
            ]
        )

        policy_network = snt.Sequential(
            [
                q_network,
                lambda q: tf.cast(
                    trfl.epsilon_greedy(q, epsilon=epsilon).sample(), "int64"
                ),
            ]
        )

        q_networks[key] = q_network
        policy_networks[key] = policy_network

    return {
        "q_networks": q_networks,
        "policies": policy_networks,
    }


def main(_: Any) -> None:

    # set loggers info
    base_dir = Path.cwd()
    log_dir = base_dir / "logs"
    log_time_stamp = str(datetime.now())

    log_info = (log_dir, log_time_stamp)

    environment_factory = lp_utils.partial_kwargs(
        make_environment, env_name=FLAGS.env_name
    )

    program = madqn.MADQN(
        environment_factory=environment_factory,
        network_factory=lp_utils.partial_kwargs(make_networks),
        num_executors=2,
        log_info=log_info,
    ).build()

    lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == "__main__":
    app.run(main)
