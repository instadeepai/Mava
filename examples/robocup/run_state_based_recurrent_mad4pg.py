# type: ignore
"""Run recurrent MA-D4PG on RoboCup."""
import functools
from datetime import datetime
from typing import Any

import launchpad as lp
from absl import app, flags
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava.components.tf.architectures import StateBasedQValueCritic
from mava.systems.tf import mad4pg
from mava.systems.tf.mad4pg.execution import MAD4PGRecurrentExecutor
from mava.systems.tf.mad4pg.training import MAD4PGStateBasedRecurrentTrainer
from mava.utils import lp_utils
from mava.utils.environments import robocup_utils
from mava.utils.environments.RoboCup_env.robocup_agents.nn_agent import (
    make_recurrent_networks as make_networks,
)
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
    # Environment
    # Create an environment, grab the spec, and use it to create networks.

    # Neural Networks
    # Create the networks
    environment_factory = lp_utils.partial_kwargs(robocup_utils.make_environment)
    network_factory = lp_utils.partial_kwargs(make_networks)

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_dir = f"{FLAGS.base_dir}/robocup"

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
