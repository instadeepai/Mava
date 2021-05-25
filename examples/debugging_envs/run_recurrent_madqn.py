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

import functools
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import launchpad as lp
import sonnet as snt
import tensorflow as tf
from absl import app, flags
from acme import types
from acme.tf import networks
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava import specs as mava_specs
from mava.components.tf.modules.exploration import LinearExplorationScheduler
from mava.components.tf.networks import epsilon_greedy_action_selector
from mava.systems.tf import madqn
from mava.systems.tf.madqn.execution import MADQNRecurrentExecutor
from mava.systems.tf.madqn.training import MADQNRecurrentTrainer
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
    q_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        512,
        512,
        256,
    ),
    shared_weights: bool = True,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(q_networks_layer_sizes, Sequence):
        q_networks_layer_sizes = {key: q_networks_layer_sizes for key in specs.keys()}

    def action_selector_fn(
        q_values: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: Optional[tf.Variable] = None,
    ) -> types.NestedTensor:
        return epsilon_greedy_action_selector(
            action_values=q_values, legal_actions_mask=legal_actions, epsilon=epsilon
        )

    q_networks = {}
    action_selectors = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the policy network.
        q_network = snt.DeepRNN(
            [
                networks.LayerNormMLP(
                    q_networks_layer_sizes[key], activate_final=False
                ),
                snt.LSTM(25),
                snt.nets.MLP([128]),
                networks.NearZeroInitializedLinear(num_dimensions),
            ]
        )

        # epsilon greedy action selector
        action_selector = action_selector_fn

        q_networks[key] = q_network
        action_selectors[key] = action_selector

    return {
        "q_networks": q_networks,
        "action_selectors": action_selectors,
    }


def main(_: Any) -> None:

    # environment
    environment_factory = functools.partial(
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
    program = madqn.MADQN(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=2,
        trainer_fn=MADQNRecurrentTrainer,
        executor_fn=MADQNRecurrentExecutor,
        exploration_scheduler_fn=LinearExplorationScheduler,
        epsilon_min=0.05,
        epsilon_decay=5e-4,
        optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        checkpoint_subpath=checkpoint_dir,
        n_step=1,
        batch_size=32,
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
