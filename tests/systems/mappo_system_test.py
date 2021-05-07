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

"""Tests for MAPPO."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Union

import dm_env
import launchpad as lp
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme.tf import networks
from acme.tf import utils as tf2_utils

import mava
from mava import specs as mava_specs
from mava.systems.tf import mappo
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    shared_weights: bool = True,
) -> Dict[str, snt.Module]:

    """Creates networks used by the agents."""

    # TODO handle observation networks.

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
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

        # Create the shared observation network; here simply a state-less operation.
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)

        # Note: The discrete case must be placed first as it inherits from BoundedArray.
        if isinstance(specs[key].actions, dm_env.specs.DiscreteArray):  # discreet
            num_actions = specs[key].actions.num_values
            policy_network = snt.Sequential(
                [
                    networks.LayerNormMLP(
                        tuple(policy_networks_layer_sizes[key]) + (num_actions,),
                        activate_final=False,
                    ),
                    tf.keras.layers.Lambda(
                        lambda logits: tfp.distributions.Categorical(logits=logits)
                    ),
                ]
            )
        elif isinstance(specs[key].actions, dm_env.specs.BoundedArray):  # continuous
            num_actions = np.prod(specs[key].actions.shape, dtype=int)
            policy_network = snt.Sequential(
                [
                    networks.LayerNormMLP(
                        policy_networks_layer_sizes[key], activate_final=True
                    ),
                    networks.MultivariateNormalDiagHead(num_dimensions=num_actions),
                    networks.TanhToSpec(specs[key].actions),
                ]
            )
        else:
            raise ValueError(f"Unknown action_spec type, got {specs[key].actions}.")

        critic_network = snt.Sequential(
            [
                networks.LayerNormMLP(
                    critic_networks_layer_sizes[key], activate_final=True
                ),
                networks.NearZeroInitializedLinear(1),
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


class TestMAPPO:
    """Simple integration/smoke test for MAPPO."""

    def test_mappo_on_debugging_env(self) -> None:
        """Tests that the system can run on the simple spread
        debugging environment without crashing."""

        # set loggers info
        base_dir = Path.cwd()
        log_dir = base_dir / "logs"
        log_time_stamp = str(datetime.now())

        log_info = (log_dir, log_time_stamp)

        # environment
        environment_factory = lp_utils.partial_kwargs(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="continuous",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(make_networks)

        # system
        system = mappo.MAPPO(
            environment_factory=environment_factory,
            network_factory=network_factory,
            log_info=log_info,
            num_executors=2,
            batch_size=32,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        lp.launch(program, launch_type="test_mt")

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(5):
            trainer.step()
