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

"""Example running recurrent MADDPG on the debug MPE environments."""
import functools
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence, Union

import launchpad as lp
import numpy as np
import sonnet as snt
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava import specs as mava_specs
from mava.components.tf.modules.communication import BroadcastedCommunication
from mava.components.tf.networks import CommunicationNetwork
from mava.systems.tf import maddpg
from mava.systems.tf.maddpg.execution import MADDPGRecurrentCommExecutor
from mava.systems.tf.maddpg.training import MADDPGDecentralisedRecurrentCommTrainer
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "continuous",
    "Environment action space type (str).",
)
flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava/", "Base dir to store experiments.")


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        32,
        32,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (128, 128, 64),
    shared_weights: bool = True,
    message_size: int = 4,
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

        # Create the observation network.
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)

        # Create the policy network.
        policy_network = CommunicationNetwork(
            snt.Sequential(
                [
                    observation_network,
                    snt.Flatten(),
                    snt.nets.MLP(policy_networks_layer_sizes[key]),
                ]
            ),
            networks.LayerNormMLP(
                (32,),
                activate_final=True,
            ),
            snt.LSTM(16),
            snt.Sequential(
                [
                    snt.nets.MLP([32]),
                    networks.NearZeroInitializedLinear(num_dimensions),
                    networks.TanhToSpec(specs[key].actions),
                    networks.ClippedGaussian(sigma),
                    networks.ClipToSpec(specs[key].actions),
                ]
            ),
            snt.Sequential(
                [
                    networks.LayerNormMLP((32, message_size), activate_final=True),
                ]
            ),
            message_size=message_size,
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
        "observations": observation_networks,
        "policies": policy_networks,
        "critics": critic_networks,
    }


def main(_: Any) -> None:

    # environment
    environment_factory = lp_utils.partial_kwargs(
        debugging_utils.make_environment,
        env_name=FLAGS.env_name,
        action_space=FLAGS.action_space,
    )

    network_factory = lp_utils.partial_kwargs(make_networks)

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_dir = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # loggers
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )

    program = maddpg.MADDPG(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=2,
        trainer_fn=MADDPGDecentralisedRecurrentCommTrainer,
        executor_fn=MADDPGRecurrentCommExecutor,
        communication_function=BroadcastedCommunication,
        checkpoint_subpath=checkpoint_dir,
    ).build()

    # launch
    gpu_id = -1
    env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    local_resources = {
        "trainer": [],
        "evaluator": PythonProcess(env=env_vars),
        "executor": PythonProcess(env=env_vars),
    }
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
