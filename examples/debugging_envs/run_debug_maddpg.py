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

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import dm_env
import launchpad as lp
import numpy as np
import sonnet as snt
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.systems.tf import maddpg
from mava.utils import lp_utils
from mava.utils.debugging.make_env import make_debugging_env
from mava.wrappers.debugging_envs import DebuggingEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes to run for.")

flags.DEFINE_integer(
    "num_episodes_per_eval",
    100,
    "Number of training episodes to run between evaluation " "episodes.",
)


def make_environment(
    evaluation: bool = False,
    env_name: str = "simple_spread",
    action_space: str = "continuous",
    num_agents: int = 3,
    render: bool = False,
) -> dm_env.Environment:

    del evaluation

    assert action_space == "continuous" or action_space == "discrete"

    """Creates a MPE environment."""
    env_module = make_debugging_env(env_name, action_space, num_agents)
    environment = DebuggingEnvWrapper(env_module, render=render)
    return environment


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    shared_weights: bool = True,
    sigma: float = 0.3,
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
    critic_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(specs[key].actions.shape, dtype=int)

        # Create the shared observation network; here simply a state-less operation.
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)

        # Create the policy network.
        policy_network = snt.Sequential(
            [
                observation_network,
                networks.LayerNormMLP(
                    policy_networks_layer_sizes[key], activate_final=True
                ),
                networks.NearZeroInitializedLinear(num_dimensions),
                networks.TanhToSpec(specs[key].actions),
                networks.ClippedGaussian(sigma),
                networks.ClipToSpec(specs[key].actions),
            ]
        )

        # Create the critic network.
        critic_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
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

    return {
        "policies": policy_networks,
        "critics": critic_networks,
        "observations": observation_networks,
    }


def main(_: Any) -> None:

    # set loggers info
    base_dir = Path.cwd()
    log_dir = base_dir / "logs"
    log_time_stamp = str(datetime.now())

    log_info = (log_dir, log_time_stamp)

    environment_factory = lp_utils.partial_kwargs(make_environment)

    program = maddpg.MADDPG(
        environment_factory=environment_factory,
        network_factory=lp_utils.partial_kwargs(make_networks),
        num_executors=2,
        log_info=log_info,
    ).build()

    (trainer_node,) = program.groups["trainer"]

    trainer_node.disable_run()

    lp.launch(program, launch_type="test_mt")

    trainer = trainer_node.create_handle().dereference()

    for _ in range(5):
        trainer.step()

    # print(dir(trainer.step()))

    # lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == "__main__":
    app.run(main)
