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

"""Example running MAPPO on pettingzoo environment."""

import functools
from datetime import datetime
from typing import Any, Dict, Sequence, Union

import acme.tf.networks as networks
import dm_env
import launchpad as lp
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from absl import app, flags
from acme.tf import utils as tf2_utils
from launchpad.nodes.python.local_multi_processing import PythonProcess
from supersuit import black_death_v1

import mava.specs as mava_specs
from mava.systems.tf import mappo
from mava.utils import lp_utils
from mava.utils.environments import pettingzoo_utils
from mava.utils.loggers import Logger

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "multiwalker_v7",
    "Debugging environment name (str).",
)

flags.DEFINE_string(
    "env_class",
    "sisl",
    "Pettingzoo environment class, e.g. atari (str).",
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
        256,
        256,
        256,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    shared_weights: bool = True,
) -> Dict[str, snt.Module]:

    """Creates networks used by the agents."""

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
        observation_network = tf2_utils.to_sonnet_module(tf.identity)

        # Note: The discrete case must be placed first as it inherits from BoundedArray.
        if isinstance(specs[key].actions, dm_env.specs.DiscreteArray):  # discrete
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


def main(_: Any) -> None:

    # environment
    environment_factory = functools.partial(
        pettingzoo_utils.make_environment,
        env_class=FLAGS.env_class,
        env_name=FLAGS.env_name,
        env_preprocess_wrappers=[(black_death_v1, None)],
    )

    # networks
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

    # distributed program
    program = mappo.MAPPO(
        environment_factory=environment_factory,
        network_factory=network_factory,
        num_executors=2,
        policy_optimizer=snt.optimizers.Adam(learning_rate=5e-4),
        critic_optimizer=snt.optimizers.Adam(learning_rate=1e-5),
        checkpoint_subpath=checkpoint_dir,
        trainer_logger=trainer_logger,
        exec_logger=exec_logger,
        eval_logger=eval_logger,
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
