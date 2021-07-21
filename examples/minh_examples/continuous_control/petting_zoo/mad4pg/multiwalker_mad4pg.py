"""Example running MAD4PG on PZ environments."""

import functools
from datetime import datetime
from typing import Any

import launchpad as lp
import sonnet as snt
from absl import app, flags
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava.systems.tf import mad4pg
from mava.utils import lp_utils
from mava.utils.environments import pettingzoo_utils
from mava.utils.loggers import logger_utils
from mava.wrappers.environment_loop_wrappers import MonitorParallelEnvironmentLoop

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "env_class",
    "sisl",
    "Pettingzoo environment class, e.g. atari (str).",
)

flags.DEFINE_string(
    "env_name",
    "multiwalker_v7",
    "Pettingzoo environment name, e.g. pong (str).",
)
flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava/multi_walker_mad4pg", "Base dir to store experiments.")


def main(_: Any) -> None:

    # Environment.
    environment_factory = functools.partial(
        pettingzoo_utils.make_environment,
        env_name=FLAGS.env_name,
        env_class=FLAGS.env_class,
        remove_on_fall=False,
    )

    # Networks.
    network_factory = lp_utils.partial_kwargs(mad4pg.make_default_networks)

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

    # Distributed program.
    program = mad4pg.MAD4PG(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=1,
        policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        checkpoint_subpath=checkpoint_dir,
        max_gradient_norm=10.0,
        
        eval_loop_fn=MonitorParallelEnvironmentLoop,
        eval_loop_fn_kwargs={"path": checkpoint_dir, "record_every": 100},
    ).build()

    # Ensure only trainer runs on gpu, while other processes run on cpu.
    gpu_id = -1
    env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    local_resources = {
        "trainer": [],
        "evaluator": PythonProcess(env=env_vars),
        "executor": PythonProcess(env=env_vars),
    }

    # Launch.
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal", # output_to_files
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)