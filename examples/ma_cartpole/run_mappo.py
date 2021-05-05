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

"""Example running MAPPO on multi-agent CartPole."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import acme.tf.networks as networks
import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from absl import app, flags

import mava.specs as mava_specs
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import mappo
from mava.utils.debugging.make_env import make_debugging_env
from mava.utils.loggers import Logger
from mava.wrappers import DetailedPerAgentStatistics
from mava.wrappers.debugging_envs import DebuggingEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes to run for.")


def make_environment(
    env_name: str = "simple_spread",
    action_space: str = "continuous",
    num_agents: int = 3,
    render: bool = False,
) -> dm_env.Environment:

    assert action_space == "continuous" or action_space == "discrete"

    """Creates a MPE environment."""
    env_module = make_debugging_env(env_name, action_space, num_agents)
    environment = DebuggingEnvWrapper(env_module, render=render)
    return environment


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    layer_sizes: Tuple = (100,),
    recurrent: bool = False,
    recurrent_layer_size: int = 20,
    shared_weights: bool = False,
) -> Dict[str, snt.Module]:

    """Creates networks used by the agents."""

    # TODO handle observation networks.

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    all_networks: Dict = {"policies": {}, "critics": {}}
    for agent_type, spec in specs.items():

        policy_network_layers = []
        critic_network_layers = []

        critic_network_layers += [
            networks.LayerNormMLP(layer_sizes, activate_final=True),
            networks.NearZeroInitializedLinear(1),
        ]

        # Note: The discrete case must be placed first as it inherits from BoundedArray.
        if isinstance(spec.actions, dm_env.specs.DiscreteArray):  # discreet
            num_actions = spec.actions.num_values
            policy_network_layers += [
                networks.LayerNormMLP(
                    layer_sizes + (num_actions,), activate_final=False
                ),
                tf.keras.layers.Lambda(
                    lambda logits: tfp.distributions.Categorical(logits=logits)
                ),
            ]
        elif isinstance(spec.actions, dm_env.specs.BoundedArray):  # continuous
            num_actions = np.prod(spec.actions.shape, dtype=int)
            policy_network_layers += [
                networks.LayerNormMLP(layer_sizes, activate_final=True),
                networks.MultivariateNormalDiagHead(num_dimensions=num_actions),
                networks.TanhToSpec(spec.actions),
            ]
        else:
            raise ValueError(f"Unknown action_spec type, got {spec.actions}.")

        if recurrent:
            policy_network_layers = [
                snt.LSTM(recurrent_layer_size)
            ] + policy_network_layers
            critic_network_layers = [
                snt.LSTM(recurrent_layer_size)
            ] + critic_network_layers

            policy_network = snt.DeepRNN(policy_network_layers)
            critic_network = snt.DeepRNN(critic_network_layers)
        else:
            policy_network = snt.Sequential(policy_network_layers)
            critic_network = snt.Sequential(critic_network_layers)

        all_networks["policies"][agent_type] = policy_network
        all_networks["critics"][agent_type] = critic_network

    return all_networks


def main(_: Any) -> None:

    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment()
    environment_spec = mava_specs.MAEnvironmentSpec(environment)
    all_networks = make_networks(environment_spec, recurrent=False)

    # Create tf loggers.
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
    system = mappo.MAPPO(
        environment_spec=environment_spec,
        networks=all_networks,
        sequence_length=5,
        sequence_period=1,
        entropy_cost=0.0,
        policy_learning_rate=1e-4,
        shared_weights=False,
        logger=system_logger,
    )

    # Create the environment loop used for training.
    train_loop = ParallelEnvironmentLoop(
        environment, system, logger=train_logger, label="train_loop"
    )

    # Wrap training loop to compute and log detailed running statistics.
    train_loop = DetailedPerAgentStatistics(train_loop)

    train_loop.run(num_episodes=FLAGS.num_episodes)


if __name__ == "__main__":
    app.run(main)
