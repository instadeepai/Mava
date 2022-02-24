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
"""Example running QMIX on SMAC"""


import functools
from datetime import datetime
from typing import Any

import launchpad as lp
import sonnet as snt
from absl import app, flags

from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationTimestepScheduler,
)
from mava.systems.tf import value_decomposition
from mava.utils import lp_utils
from mava.utils.environments.smac_utils import make_environment
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "map_name",
    "3m",
    "Starcraft 2 micromanagement map name (str).",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)

flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def main(_: Any) -> None:
    """Example running recurrent QMIX on SMAC environment."""

    # Environment
    environment_factory = functools.partial(make_environment, map_name=FLAGS.map_name)

    # Networks.
    network_factory = lp_utils.partial_kwargs(
        value_decomposition.make_default_networks,
    )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_dir = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )

    # Distributed program
    program = value_decomposition.ValueDecomposition(
        environment_factory=environment_factory,
        network_factory=network_factory,
        mixer="qmix",
        logger_factory=logger_factory,
        num_executors=1,
        exploration_scheduler_fn=LinearExplorationTimestepScheduler(
            epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_steps=50000
        ),
        optimizer=snt.optimizers.RMSProp(
            learning_rate=0.0005, epsilon=0.00001, decay=0.99
        ),
        checkpoint_subpath=checkpoint_dir,
        batch_size=32,
        executor_variable_update_period=200,
        target_update_period=200,
        max_gradient_norm=20.0,
        min_replay_size=32,
        max_replay_size=5000,
        samples_per_insert=4,
        sequence_length=20,
        period=10,
        evaluator_interval={"executor_steps": 2000},
    ).build()

    # Only the trainer should use the GPU (if available)
    local_resources = lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=["trainer"]
    )

    # Launch
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
