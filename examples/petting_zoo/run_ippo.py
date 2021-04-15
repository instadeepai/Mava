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

"""Example running IPPO on pettinzoo MPE environments."""

import importlib
from typing import Any, Dict, Mapping, Sequence, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils.loggers.tf_summary import TFSummaryLogger

from mava import specs as mava_specs
from mava.environment_loops.pettingzoo import PettingZooParallelEnvironmentLoop
from mava.systems.tf import ippo
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 100, "Number of training episodes to run for.")

flags.DEFINE_integer(
    "num_episodes_per_eval",
    10,
    "Number of training episodes to run between evaluation " "episodes.",
)


def make_environment(
    env_class: str = "mpe", env_name: str = "simple_v2", **kwargs: int
) -> dm_env.Environment:
    """Creates a MPE environment."""
    env_module = importlib.import_module(f"pettingzoo.{env_class}.{env_name}")
    env = env_module.parallel_env(**kwargs)  # type: ignore
    environment = PettingZooParallelEnvWrapper(env)
    return environment


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256, 256),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256, 256),
    shared_weights: bool = True,
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
                # The multiplexer concatenates the observations/actions.
                # TODO : seek a around not using actions here.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(
                    critic_networks_layer_sizes[key], activate_final=False
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
    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment()
    environment_spec = mava_specs.MAEnvironmentSpec(environment)
    system_networks = make_networks(environment_spec)

    # create tf loggers
    logs_dir = "logs"
    system_logger = TFSummaryLogger(f"{logs_dir}/system")
    train_logger = TFSummaryLogger(f"{logs_dir}/train_loop")
    # eval_logger = TFSummaryLogger(f"{logs_dir}/eval_loop")

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
    )

    # Create the environment loop used for training.
    train_loop = PettingZooParallelEnvironmentLoop(
        environment, system, logger=train_logger, label="train_loop"
    )

    # Create the evaluation policy.
    # NOTE: assumes weight sharing
    # specs = environment_spec.get_agent_specs()
    # type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
    # specs = type_specs
    # eval_policies = {
    #     key: snt.Sequential(
    #         [
    #             system_networks["observations"][key],
    #             system_networks["policies"][key],
    #         ]
    #     )
    #     for key in specs.keys()
    # }

    # # Create the evaluation actor and loop.
    # eval_actor = executors.FeedForwardExecutor(policy_networks=eval_policies)
    # eval_env = make_environment(remove_on_fall=False)
    # eval_loop = PettingZooParallelEnvironmentLoop(
    #     eval_env, eval_actor, logger=eval_logger, label="eval_loop"
    # )

    for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
        train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
        # eval_loop.run(num_episodes=1)
        print("hi")


if __name__ == "__main__":
    app.run(main)
