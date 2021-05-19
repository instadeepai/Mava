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

"""Example running DIAL"""
import functools
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence, Union

import launchpad as lp
import numpy as np
import tensorflow as tf
from absl import app, flags
from acme import types
from acme.tf import utils as tf2_utils
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym import spaces

from mava import specs as mava_specs
from mava.components.tf.networks import DIALPolicy
from mava.systems.tf import dial
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 30000, "Number of training episodes to run for.")
flags.DEFINE_string(
    "env_name",
    "switch",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "discrete",
    "Environment action space type (str).",
)
flags.DEFINE_integer(
    "num_episodes_per_eval",
    100,
    "Number of training episodes to run between evaluation " "episodes.",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava/", "Base dir to store experiments.")


# TODO Kevin: Define message head node correctly
def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_network_gru_hidden_sizes: Union[Dict[str, int], int] = 128,
    policy_network_gru_layers: Union[Dict[str, int], int] = 2,
    policy_network_task_mlp_sizes: Union[Dict[str, Sequence], Sequence] = (128,),
    policy_network_message_in_mlp_sizes: Union[Dict[str, Sequence], Sequence] = (128,),
    policy_network_message_out_mlp_sizes: Union[Dict[str, Sequence], Sequence] = (128,),
    policy_network_output_mlp_sizes: Union[Dict[str, Sequence], Sequence] = (128,),
    message_size: int = 1,
    shared_weights: bool = True,
    sigma: float = 0.3,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()
    # extra_specs = environment_spec.get_extra_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(policy_network_gru_hidden_sizes, int):
        policy_network_gru_hidden_sizes = {
            key: policy_network_gru_hidden_sizes for key in specs.keys()
        }

    if isinstance(policy_network_gru_layers, int):
        policy_network_gru_layers = {
            key: policy_network_gru_layers for key in specs.keys()
        }

    if isinstance(policy_network_task_mlp_sizes, Sequence):
        policy_network_task_mlp_sizes = {
            key: policy_network_task_mlp_sizes for key in specs.keys()
        }

    if isinstance(policy_network_message_in_mlp_sizes, Sequence):
        policy_network_message_in_mlp_sizes = {
            key: policy_network_message_in_mlp_sizes for key in specs.keys()
        }

    if isinstance(policy_network_message_out_mlp_sizes, Sequence):
        policy_network_message_out_mlp_sizes = {
            key: policy_network_message_out_mlp_sizes for key in specs.keys()
        }

    if isinstance(policy_network_output_mlp_sizes, Sequence):
        policy_network_output_mlp_sizes = {
            key: policy_network_output_mlp_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}

    for key in specs.keys():
        # Create the shared observation network
        observation_network = tf2_utils.to_sonnet_module(tf.identity)

        # Create the policy network.
        policy_network = DIALPolicy(
            action_spec=specs[key].actions,
            message_spec=_convert_to_spec(
                spaces.Box(-np.inf, np.inf, (message_size,), dtype=np.float32)
            ),
            gru_hidden_size=policy_network_gru_hidden_sizes[key],
            gru_layers=policy_network_gru_layers[key],
            task_mlp_size=policy_network_task_mlp_sizes[key],
            message_in_mlp_size=policy_network_message_in_mlp_sizes[key],
            message_out_mlp_size=policy_network_message_out_mlp_sizes[key],
            output_mlp_size=policy_network_output_mlp_sizes[key],
        )

        observation_networks[key] = observation_network
        policy_networks[key] = policy_network

    return {
        "policies": policy_networks,
        "observations": observation_networks,
    }


def main(_: Any) -> None:
    # set loggers info
    log_info = (FLAGS.base_dir, f"{FLAGS.mava_id}/logs")

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

    program = dial.DIAL(
        environment_factory=environment_factory,
        network_factory=network_factory,
        num_executors=5,
        batch_size=1,
        log_info=log_info,
        checkpoint_subpath=checkpoint_dir,
        trainer_logger=trainer_logger,
        exec_logger=exec_logger,
        eval_logger=eval_logger,
    ).build()

    lp.launch(
        program, lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal="current_terminal"
    )


if __name__ == "__main__":
    app.run(main)
