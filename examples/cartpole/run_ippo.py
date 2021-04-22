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

"""Example running IPPO on MA CartPole environment."""

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils
from ray.rllib.env.multi_agent_env import make_multi_agent

from mava import specs as mava_specs
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import ippo
from mava.utils.loggers import Logger

# from mava.systems.tf.ippo import execution as executors
from mava.wrappers import DetailedPerAgentStatistics, RLLibMultiAgentEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 5000, "Number of training episodes to run for.")

flags.DEFINE_integer(
    "num_episodes_per_eval",
    10,
    "Number of training episodes to run between evaluation " "episodes.",
)


def make_environment(
    env_name: str = "CartPole-v1", num_envs: int = 2, **kwargs: int
) -> dm_env.Environment:
    """Creates a MA-CartPole environment."""
    ma_cls = make_multi_agent(env_name)
    ma_env = ma_cls({"num_agents": num_envs})
    wrapped_env = RLLibMultiAgentEnvWrapper(ma_env)
    return wrapped_env


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256, 256),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256, 256),
    shared_weights: bool = False,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_layer_sizes, Sequence):
        critic_networks_layer_sizes = {
            key: critic_networks_layer_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    behavior_networks = {}
    critic_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(specs[key].actions.num_values, dtype=int)

        # Create the shared observation network
        observation_network = tf2_utils.to_sonnet_module(tf.identity)

        # Create the policy network.
        policy_network = snt.Sequential(
            [
                networks.LayerNormMLP(
                    policy_networks_layer_sizes[key], activate_final=True
                ),
                networks.NearZeroInitializedLinear(num_dimensions),
            ]
        )

        # Create the behavior policy.
        behavior_network = snt.Sequential([observation_network, policy_network])

        # Create the critic network.
        critic_network = snt.Sequential(
            [
                networks.LayerNormMLP(
                    critic_networks_layer_sizes[key], activate_final=True
                ),
                snt.Linear(1),
            ]
        )
        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_networks[key] = critic_network
        behavior_networks[key] = behavior_network
    return {
        "policies": policy_networks,
        "critics": critic_networks,
        "observations": observation_networks,
        "behaviors": behavior_networks,
    }


def main(_: Any) -> None:
    shared_weights = False
    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment()
    environment_spec = mava_specs.MAEnvironmentSpec(environment)
    system_networks = make_networks(environment_spec, shared_weights=shared_weights)

    # create tf loggers
    base_dir = Path.cwd()
    log_dir = base_dir / "logs"
    log_time_stamp = str(datetime.now())

    system_logger = Logger(
        label="system_trainer",
        directory=log_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=log_time_stamp,
    )
    train_logger = Logger(
        label="train_loop",
        directory=log_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=log_time_stamp,
    )

    # Construct the agent.
    system = ippo.IPPO(
        environment_spec=environment_spec,
        policy_networks=system_networks["policies"],
        critic_networks=system_networks["critics"],
        observation_networks=system_networks[
            "observations"
        ],  # pytype: disable=wrong-arg-types
        behavior_networks=system_networks["behaviors"],
        logger=system_logger,
        max_queue_size=10,
        batch_size=5,
        shared_weights=shared_weights,
    )

    # Create the environment loop used for training.
    train_loop = ParallelEnvironmentLoop(
        environment, system, logger=train_logger, label="train_loop"
    )

    # Wrap training loop to compute and log detailed running statistics
    train_loop = DetailedPerAgentStatistics(train_loop)

    # system._trainer.run()
    for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
        train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)


if __name__ == "__main__":
    app.run(main)
