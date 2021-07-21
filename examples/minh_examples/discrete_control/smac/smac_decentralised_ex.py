# Script run Multi-Agent DQN systems Feedforward decentralised on SMAC discrete control

"""Example running MADQN on multi-agent Starcraft 2 (SMAC) environment."""
import functools
from datetime import datetime
from typing import Any

import launchpad as lp
import sonnet as snt
from absl import app, flags
from launchpad.nodes.python.local_multi_processing import PythonProcess


from mava.components.tf.modules.exploration import LinearExplorationScheduler
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.environments import smac_utils
from mava.utils.loggers import logger_utils
from mava.wrappers import MonitorParallelEnvironmentLoop

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
flags.DEFINE_string("base_dir", "~/mava/checkpoints_smac", "Base dir to store experiments.")


def main(_: Any) -> None:

    # Environment
    environment_factory = functools.partial(
        smac_utils.make_environment, map_name=FLAGS.map_name
    )

    # Networks.
    network_factory = lp_utils.partial_kwargs(
        madqn.make_default_networks, policy_networks_layer_sizes=[64, 64]
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

    # distributed program
    program = madqn.MADQN(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=1,
        exploration_scheduler_fn=LinearExplorationScheduler,
        epsilon_min=0.05,
        epsilon_decay=1e-5,
        optimizer=snt.optimizers.SGD(learning_rate=1e-2),
        checkpoint_subpath=checkpoint_dir,
        batch_size=512,
        executor_variable_update_period=100,
        target_update_period=200,
        max_gradient_norm=10.0,

        # Record agents in environment. 
        eval_loop_fn=MonitorParallelEnvironmentLoop,
        eval_loop_fn_kwargs={"path": checkpoint_dir, "record_every": 10, "fps": 5},
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
        terminal="current_terminal", # "current_terminal" output_to_files
        local_resources=local_resources,
    )
    

if __name__ == "__main__":
    app.run(main)