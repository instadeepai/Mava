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
"""Run feedforward IPPO on SMAC using best checkpointed networks."""


import functools
import time
from datetime import datetime
from typing import Any

import optax
from absl import app, flags

from mava.systems import ippo
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

# Used for checkpoints, tensorboard logging and env monitoring
experiment_path = f"{FLAGS.base_dir}/{FLAGS.mava_id}"


def run_system() -> None:
    """Example running feedforward IPPO on SMAC environment."""

    # Environment
    environment_factory = functools.partial(make_environment, map_name=FLAGS.map_name)

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=(64, 64),
            critic_layer_sizes=(64, 64, 64),
            *args,
            **kwargs,
        )

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
        evaluation_interval={"executor_steps": 2000},
        evaluation_duration={"evaluator_episodes": 32},
        executor_parameter_update_period=1,
        # Flag to activate best checkpointing
        checkpoint_best_perf=True,
        # metrics to checkpoint its best performance networks
        checkpointing_metric=("mean_episode_return", "win_rate"),
        termination_condition={"executor_steps": 30000},
        checkpoint_minute_interval=1,
        wait=True,
    )

    # Launch the system.
    system.launch()


def run_checkpointed_model() -> None:
    """Run IPPO with restored networks."""

    # Environment
    environment_factory = functools.partial(make_environment, map_name=FLAGS.map_name)

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=(64, 64),
            critic_layer_sizes=(64, 64, 64),
            *args,
            **kwargs,
        )

    # Use same dir to restore checkpointed params
    old_experiment_path = experiment_path

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
        experiment_path=old_experiment_path,
        policy_optimiser=policy_optimiser,
        critic_optimiser=critic_optimiser,
        run_evaluator=True,
        sample_batch_size=5,
        num_epochs=15,
        num_executors=1,
        multi_process=True,
        evaluation_interval={"executor_steps": 2000},
        evaluation_duration={"evaluator_episodes": 32},
        executor_parameter_update_period=5,
        # choose which metric you want to restore its best netrworks
        restore_best_net="win_rate",
        termination_condition={"executor_steps": 40000},
        checkpoint_minute_interval=1,
        wait=True,
    )

    # Launch the system.
    system.launch()


def main(_: Any) -> None:
    """Run the model and then restore the best networks"""
    # Run system that checkpoint the best performance for win rate
    # and mean return
    run_system()
    print("Start restored win rate best networks")
    time.sleep(10)
    run_checkpointed_model()


if __name__ == "__main__":
    app.run(main)
