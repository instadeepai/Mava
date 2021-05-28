# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

"""Tests for MASAC."""

import functools
from typing import Dict, Mapping, Sequence, Union

import launchpad as lp
import numpy as np
import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils
from launchpad.nodes.python.local_multi_processing import PythonProcess

import mava
from mava import specs as mava_specs
from mava.components.tf import architectures
from mava.components.tf.networks import ActorNetwork
from mava.systems.tf import masac
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_V_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        512,
        512,
        256,
    ),
    critic_Q_1_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        512,
        512,
        256,
    ),
    critic_Q_2_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        512,
        512,
        256,
    ),
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
    if isinstance(critic_V_networks_layer_sizes, Sequence):
        critic_V_networks_layer_sizes = {
            key: critic_V_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_Q_1_networks_layer_sizes, Sequence):
        critic_Q_1_networks_layer_sizes = {
            key: critic_Q_1_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_Q_2_networks_layer_sizes, Sequence):
        critic_Q_2_networks_layer_sizes = {
            key: critic_Q_2_networks_layer_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    critic_V_networks = {}
    critic_Q_1_networks = {}
    critic_Q_2_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(specs[key].actions.shape, dtype=int)

        # Create the shared observation network; here simply a state-less operation.
        # observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)
        observation_network = tf2_utils.to_sonnet_module(tf.identity)

        # Create the policy network.
        policy_network = ActorNetwork(
            256,
            256,
            256,
            num_dimensions,
            1e-6,
            observation_network,
            is_deterministic=False,
        )

        # Create the critic network.
        critic_V_network = snt.Sequential(
            [
                networks.LayerNormMLP(
                    critic_V_networks_layer_sizes[key], activate_final=False
                ),
                snt.Linear(1),
            ]
        )

        # Create the critic network.
        critic_Q_1_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(
                    critic_Q_1_networks_layer_sizes[key], activate_final=False
                ),
                snt.Linear(1),
            ]
        )

        # Create the critic network.
        critic_Q_2_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(
                    critic_Q_2_networks_layer_sizes[key], activate_final=False
                ),
                snt.Linear(1),
            ]
        )
        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_V_networks[key] = critic_V_network
        critic_Q_1_networks[key] = critic_Q_1_network
        critic_Q_2_networks[key] = critic_Q_2_network

    return {
        "policies": policy_networks,
        "critics_V": critic_V_networks,
        "critics_Q_1": critic_Q_1_networks,
        "critics_Q_2": critic_Q_2_networks,
        "observations": observation_networks,
    }


class TestMASAC:
    """Simple integration/smoke test for MASAC."""

    def test_masac_on_debugging_env(self) -> None:
        """Tests that the system can run on the simple spread
        debugging environment without crashing."""

        # set loggers info
        # TODO Allow for no checkpointing and no loggers to be
        # passed in.
        mava_id = "tests/masac"
        base_dir = "~/mava"

        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="continuous",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(make_networks)

        # system
        checkpoint_dir = f"{base_dir}/{mava_id}"

        # system
        system = masac.MASAC(
            environment_factory=environment_factory,
            network_factory=network_factory,
            architecture=architectures.CentralisedSoftQValueCritic,
            num_executors=2,
            batch_size=32,
            min_replay_size=32,
            max_replay_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_V_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_Q_1_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_Q_2_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            checkpoint=False,
            checkpoint_subpath=checkpoint_dir,
            # shared_weights=False,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        # Launch gpu config - don't use gpu
        gpu_id = -1
        env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
        local_resources = {
            "trainer": PythonProcess(env=env_vars),
            "evaluator": PythonProcess(env=env_vars),
            "executor": PythonProcess(env=env_vars),
        }

        lp.launch(
            program,
            launch_type="test_mt",
            local_resources=local_resources,
        )

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(5):
            trainer.step()
