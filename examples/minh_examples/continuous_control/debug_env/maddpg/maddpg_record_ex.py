"""Example running MADDPG on debug MPE environments, while recording agents."""
import functools
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence, Union

import launchpad as lp
import numpy as np
import sonnet as snt
import tensorflow as tf
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava import specs as mava_specs
from mava.systems.tf import maddpg
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from mava.wrappers import MonitorParallelEnvironmentLoop
from mava.components.tf import architectures
from mava.utils.loggers import logger_utils

def main(_: Any) -> None: 
    # Defind Agent Networks
    network_factory = lp_utils.partial_kwargs(maddpg.make_default_networks)

    # Select Envrironment
    env_name = "simple_spread"
    action_space = "continuous"

    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name=env_name,
        action_space=action_space,
    )

    # Create MARL Systems:
    # Specify logging and checkpoint for configuration:
    base_dir = "~/mava/checkpoints"

    # File name 
    mava_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Log every [log_every] seconds
    log_every = 15
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=mava_id,
        time_delta=log_every,
    )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_dir = f"{base_dir}/{mava_id}"

    # Create MADDPG System
    system = maddpg.MADDPG(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=1,
        policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        checkpoint_subpath=checkpoint_dir,
        max_gradient_norm=10.0,
        checkpoint=False,
        batch_size=1024,

        # Record agents in environment. 
        eval_loop_fn=MonitorParallelEnvironmentLoop,
        eval_loop_fn_kwargs={"path": checkpoint_dir, "record_every": 10, "fps": 5},
    ).build()

    # Run MADDPG System:
    # Ensure only trainer runs on gpu, while other processes run on cpu. 
    gpu_id = -1
    env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    local_resources = {
        "trainer": [],
        "evaluator": PythonProcess(env=env_vars),
        "executor": PythonProcess(env=env_vars),
    }

    lp.launch(
        system,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="output_to_files",
        local_resources=local_resources,
    )

if __name__ == "__main__":
    app.run(main)

