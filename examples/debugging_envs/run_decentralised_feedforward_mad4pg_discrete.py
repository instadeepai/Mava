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

"""Example running MAD4PG on debug MPE environments."""
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
from dm_env import specs
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava import specs as mava_specs
from mava.components.tf.networks.mad4pg import DiscreteValuedHead
from mava.systems.tf import mad4pg
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "discrete",
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
    agent_net_config: Dict[str, str],
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    sigma: float = 0.3,
    vmin: float = -150.0,
    vmax: float = 150.0,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    specs = {agent_net_config[key]: specs[key] for key in specs.keys()}

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
        # TODO (dries): Make specs[key].actions
        #  return a list of specs for hybrid action space
        # Get total number of action dimensions from action spec.
        agent_act_spec = specs[key].actions
        if type(specs[key].actions) == DiscreteArray:
            num_actions = agent_act_spec.num_values
            minimum = [-1.0] * num_actions
            maximum = [1.0] * num_actions
            agent_act_spec = BoundedArray(
                shape=(num_actions,),
                minimum=minimum,
                maximum=maximum,
                dtype="float32",
                name="actions",
            )

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(agent_act_spec.shape, dtype=int)

        # Create the shared observation network; here simply a state-less operation.
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)

        # Create the policy network.
        policy_network = snt.Sequential(
            [
                networks.LayerNormMLP(
                    policy_networks_layer_sizes[key], activate_final=True
                ),
                networks.NearZeroInitializedLinear(num_dimensions),
                networks.TanhToSpec(agent_act_spec),
                networks.ClippedGaussian(sigma),
                networks.ClipToSpec(agent_act_spec),
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
                DiscreteValuedHead(vmin, vmax, num_atoms),
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
    environment_factory = lp_utils.partial_kwargs(
        debugging_utils.make_environment,
        env_name=FLAGS.env_name,
        action_space=FLAGS.action_space,
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
    program = mad4pg.MAD4PG(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=2,
        policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        checkpoint_subpath=checkpoint_dir,
        max_gradient_norm=40.0,
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
