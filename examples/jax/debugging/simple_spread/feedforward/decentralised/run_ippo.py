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

"""Example running IPPO on debug MPE environments."""
import functools
import os
import signal
import time
from datetime import datetime
from typing import Any

import launchpad as lp
import optax
import psutil
from absl import app, flags

from mava.systems.jax import ippo
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
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def terminator(worker_manager):
    if worker_manager._active_workers != {}:
        if "data_server/0" in list(worker_manager._active_workers.keys()):
            if len(worker_manager._active_workers["data_server/0"]) > 0:
                print(worker_manager._active_workers["data_server/0"])
                parent_pid = worker_manager._active_workers["data_server/0"][0]._pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        if "executor/0" in list(worker_manager._active_workers.keys()):
            if len(worker_manager._active_workers["executor/0"]) > 0:
                print(worker_manager._active_workers["executor/0"])
                parent_pid = worker_manager._active_workers["executor/0"][0]._pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        if "trainer/0" in list(worker_manager._active_workers.keys()):
            if len(worker_manager._active_workers["trainer/0"]) > 0:
                print(worker_manager._active_workers["trainer/0"])
                parent_pid = worker_manager._active_workers["trainer/0"][0]._pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        if "evaluator/0" in list(worker_manager._active_workers.keys()):
            if len(worker_manager._active_workers["evaluator/0"]) > 0:
                print(worker_manager._active_workers["evaluator/0"])
                parent_pid = worker_manager._active_workers["evaluator/0"][0]._pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        if "parameter_server/0" in list(worker_manager._active_workers.keys()):
            if len(worker_manager._active_workers["parameter_server/0"]) > 0:
                print(worker_manager._active_workers["parameter_server/0"])
                parent_pid = worker_manager._active_workers["parameter_server/0"][
                    0
                ]._pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(
                    recursive=True
                ):  # or parent.children() for recursive=False
                    child.kill()
                parent.kill()


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Environment.
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name=FLAGS.env_name,
        action_space=FLAGS.action_space,
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=(256, 256, 256),
            critic_layer_sizes=(512, 512, 256),
            *args,
            **kwargs,
        )

    # Used for checkpoints, tensorboard logging and env monitoring
    experiment_path = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

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

    # Optimisers.
    policy_optimiser = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    critic_optimiser = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the system.
    system = ippo.IPPOSystem()

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=experiment_path,
        policy_optimiser=policy_optimiser,
        critic_optimiser=critic_optimiser,
        run_evaluator=True,
        sample_batch_size=5,
        num_epochs=15,
        num_executors=1,
        multi_process=True,
        clip_value=False,
        # termination_condition={"executor_steps": 5000}
    )

    # Launch the system.
    system.launch()

    worker_manager = system._builder.store.worker_manager
    time.sleep(20)
    terminator(worker_manager)
    """while terminator(worker_manager)!=True:
        time.sleep(2)
    i=0"""
    time.sleep(10)
    still_running = [
        label
        for label in worker_manager._active_workers
        if worker_manager._active_workers[label]
    ]
    print("STATE", still_running)
    print("STILL RUNNING")
    exit()


if __name__ == "__main__":
    app.run(main)
