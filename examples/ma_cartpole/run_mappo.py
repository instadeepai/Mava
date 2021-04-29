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
from typing import Any, Dict

import dm_env
import numpy as np
import sonnet as snt
from absl import app, flags
from acme.tf import networks
from ray.rllib.env.multi_agent_env import make_multi_agent

from mava import specs as mava_specs
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import mappo
from mava.utils.loggers import Logger
from mava.wrappers import DetailedPerAgentStatistics

# Testing, remove later.
from rllib_multi_env_wrapper import RLLibMultiAgentEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes to run for.")


def make_environment(
    env_name: str = "CartPole-v1", num_agents: int = 2, **kwargs: int
) -> dm_env.Environment:

    """Creates a MPE environment."""

    ma_cls = make_multi_agent(env_name)
    ma_env = ma_cls({"num_agents": num_agents})
    wrapped_env = RLLibMultiAgentEnvWrapper(ma_env)
    return wrapped_env


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    shared_weights: bool = False,
) -> Dict[str, snt.RNNCore]:

    """Creates networks used by the agents."""

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs.
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    all_networks: Dict = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_actions = np.prod(specs[key].actions.num_values, dtype=int)

        # Create the network.
        network = snt.DeepRNN(
            [
                snt.Flatten(),
                snt.LSTM(20),
                snt.nets.MLP([50]),
                networks.PolicyValueHead(num_actions),
            ]
        )

        all_networks[key] = network

    return all_networks


def main(_: Any) -> None:

    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment()
    environment_spec = mava_specs.MAEnvironmentSpec(environment)
    all_networks = make_networks(environment_spec)

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
        sequence_length=10,
        sequence_period=2,
        batch_size=64,
        baseline_cost=0.0001,
        critic_learning_rate=1e-3,
        policy_learning_rate=1e-3,
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
