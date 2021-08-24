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
"""Run recurrent MAD4PG on RoboCup."""
import functools
from datetime import datetime
from typing import Any

import launchpad as lp
from absl import app, flags

from mava.components.tf.architectures import StateBasedQValueCritic
from mava.systems.tf import mad4pg
from mava.systems.tf.mad4pg.execution import MAD4PGRecurrentExecutor
from mava.systems.tf.mad4pg.training import MAD4PGStateBasedRecurrentTrainer
from mava.utils import lp_utils
from mava.utils.enums import ArchitectureType
from mava.utils.environments import robocup_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)

flags.DEFINE_string("num_executors", "2", "The number of executors to run.")
flags.DEFINE_string("base_dir", "~/mava/", "Base dir to store experiments.")


def main(_: Any) -> None:
    # Environment.
    environment_factory = lp_utils.partial_kwargs(robocup_utils.make_environment)

    # Networks.
    network_factory = lp_utils.partial_kwargs(
        mad4pg.make_default_networks,
        archecture_type=ArchitectureType.recurrent,
        vmin=-5,
        vmax=5,
    )

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
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

    program = mad4pg.MAD4PG(
        architecture=StateBasedQValueCritic,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=int(FLAGS.num_executors),
        samples_per_insert=None,
        trainer_fn=MAD4PGStateBasedRecurrentTrainer,
        executor_fn=MAD4PGRecurrentExecutor,
        shared_weights=True,
        checkpoint_subpath=checkpoint_dir,
        batch_size=265,
    ).build()

    # launch
    local_resources = lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=["trainer"]
    )
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
