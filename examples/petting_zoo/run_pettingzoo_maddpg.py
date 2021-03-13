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
from typing import Mapping, Sequence, Union

from absl import app
from absl import flags
import acme
from acme import types
from acme import wrappers
from acme.tf import networks
from acme.tf import utils as tf2_utils
from pettinzoo.mpe import simple_spread_v2
import dm_env
import numpy as np
import sonnet as snt

from mava import specs
from mava.systems.tf import executors
from mava.systems.tf import maddpg
from mava.environment_loops.pettingzoo import PettingZooParallelEnvironmentLoop

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 100, "Number of training episodes to run for.")

flags.DEFINE_integer(
    "num_episodes_per_eval",
    10,
    "Number of training episodes to run between evaluation " "episodes.",
)


def make_environment(env_name: str = "simple_spread_v2") -> dm_env.Environment:
    """Creates a MPE environment."""
    env_module = importlib.import_module(f"pettingzoo.mpe.{env_name}")
    environment = env_module.env()
    return environment


def make_networks(
    action_specs: specs.BoundedArray,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence],
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence],
    shared_weights: bool = False,
    vmin: float = -150.0,
    vmax: float = 150.0,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in action_specs.keys()
        }
    if isinstance(critic_networks_layer_sizes, Sequence):
        critic_networks_layer_sizes = {
            key: critic_networks_layer_sizes for key in action_specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    critic_networks = {}
    for key in action_specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(action_specs[key].shape, dtype=int)

        # Create the shared observation network; here simply a state-less operation.
        observation_network = tf2_utils.batch_concat

        # Create the policy network.
        policy_network = snt.Sequential(
            [
                networks.LayerNormMLP(
                    policy_networks_layer_sizes[key], activate_final=True
                ),
                networks.NearZeroInitializedLinear(num_dimensions),
                networks.TanhToSpec(action_specs[key]),
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


def main(_):
    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment()
    environment_specs = specs.SystemSpec(environment)
    agents, agent_types = environment_specs.get_agent_info()
    system_networks = make_networks(environment_specs.spec.actions)

    # Construct the agent.
    system = maddpg.MADDPG(
        environment_spec=environment_specs.spec,
        policy_network=system_networks["policies"],
        critic_network=system_networks["critics"],
        observation_network=system_networks[
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
    eval_actor = executors.FeedForwardActor(policy_networks=eval_policies)
    eval_env = make_environment()
    eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label="eval_loop")

    for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
        train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
        eval_loop.run(num_episodes=1)


if __name__ == "__main__":
    app.run(main)
