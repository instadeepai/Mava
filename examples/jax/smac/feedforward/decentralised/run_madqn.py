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
"""Run feedforward MADQN on SMAC."""


import functools
import os
from datetime import datetime
from typing import Any

import optax
from absl import app, flags

from mava.systems.jax import madqn
from mava.utils.environments.smac_utils import make_environment
from mava.utils.loggers import logger_utils
from mava.utils.schedules.linear_epsilon_scheduler import LinearEpsilonScheduler

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

# TODO: Remove. Only needed when running on gpu, outside of docker.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def main(_: Any) -> None:
    """Example running feedforward MADQN on SMAC environment."""

    # Environment
    environment_factory = functools.partial(
        make_environment, map_name=FLAGS.map_name, return_state_info=False
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return madqn.make_default_networks(policy_layer_sizes=(64, 64), **kwargs)

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_subpath = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

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

    # Optim
    optimizer = optax.rmsprop(learning_rate=0.0005, eps=0.00001, decay=0.99)
    optimizer = optax.chain(optax.clip_by_global_norm(20.0), optimizer)

    # epsilon scheduler
    epsilon_scheduler = LinearEpsilonScheduler(1.0, 0.05, 200000)

    # Create the system.
    system = madqn.MADQNSystem()

    # Build the system.
    system.build(
        epsilon_scheduler=epsilon_scheduler,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=200,
        multi_process=True,
        run_evaluator=True,
        num_executors=1,
        # use_next_extras=False,
        sample_batch_size=256,
        target_update_period=200,
        num_epochs=5,  # TODO not used any more - remove
        min_data_server_size=1_000,
        n_step=5,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
