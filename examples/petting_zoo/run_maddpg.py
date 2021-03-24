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

"""Example running MADDPG on pettinzoo MPE environments."""

import importlib
from typing import Any, Dict, Mapping, Sequence, Union

import dm_env
import numpy as np
import sonnet as snt
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.environment_loops.pettingzoo import PettingZooParallelEnvironmentLoop
from mava.systems.tf import executors, maddpg
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 100, "Number of training episodes to run for.")

flags.DEFINE_integer(
    "num_episodes_per_eval",
    10,
    "Number of training episodes to run between evaluation " "episodes.",
)


def make_environment(
    env_type: str = "sisl", env_name: str = "multiwalker_v6"
) -> dm_env.Environment:
    """Creates a MPE environment."""
    env_module = importlib.import_module(f"pettingzoo.{env_type}.{env_name}")
    env = env_module.parallel_env()  # type: ignore
    environment = PettingZooParallelEnvWrapper(env)
    return environment


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256, 256),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    shared_weights: bool = False,
    vmin: float = -150.0,
    vmax: float = 150.0,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()
    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_layer_sizes, Sequence):
        critic_networks_layer_sizes = {
            key: critic_networks_layer_sizes for key in specs.keys()
        }

    specs = environment_spec.get_agent_specs()
    observation_networks = {}
    policy_networks = {}
    critic_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(specs[key].actions.shape, dtype=int)

        # Create the shared observation network; here simply a state-less operation.
        observation_network = tf2_utils.batch_concat

        # Create the policy network.
        policy_network = snt.Sequential(
            [
                networks.LayerNormMLP(
                    policy_networks_layer_sizes[key], activate_final=True
                ),
                networks.NearZeroInitializedLinear(num_dimensions),
                networks.TanhToSpec(specs[key].actions),
            ]
        )

        # Create the critic network.
        critic_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(
                    critic_networks_layer_sizes[key], activate_final=True
                ),
                networks.DiscreteValuedHead(vmin, vmax, num_atoms),
            ]
        )
        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_networks[key] = critic_network

    return {
        "policies": policy_networks,
        "critics": critic_networks,
        "observations": observation_networks,
    }


def main(_: Any) -> None:
    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment()
    environment_spec = mava_specs.MAEnvironmentSpec(environment)
    system_networks = make_networks(environment_spec)

    # Construct the agent.
    system = maddpg.MADDPG(
        environment_spec=environment_spec,
        policy_networks=system_networks["policies"],
        critic_networks=system_networks["critics"],
        observation_networks=system_networks[
            "observations"
        ],  # pytype: disable=wrong-arg-types
    )

    # Create the environment loop used for training.
    train_loop = PettingZooParallelEnvironmentLoop(
        environment, system, label="train_loop"
    )

    # Create the evaluation policy.
    eval_policies = snt.Sequential(
        [
            system_networks["observations"],
            system_networks["policies"],
        ]
    )

    # Create the evaluation actor and loop.
    eval_actor = executors.FeedForwardExecutor(policy_networks=eval_policies)
    eval_env = make_environment()
    eval_loop = PettingZooParallelEnvironmentLoop(
        eval_env, eval_actor, label="eval_loop"
    )

    for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
        train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
        eval_loop.run(num_episodes=1)


if __name__ == "__main__":
    app.run(main)
