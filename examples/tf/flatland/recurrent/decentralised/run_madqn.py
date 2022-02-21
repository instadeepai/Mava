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
"""Example running recurrent MADQN on Flatland."""

import functools
from datetime import datetime
from typing import Any, Dict

import launchpad as lp
from absl import app, flags

from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.enums import ArchitectureType
from mava.utils.environments.flatland_utils import make_environment
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


# flatland environment config
env_config: Dict = {
    "n_agents": 10,
    "x_dim": 30,
    "y_dim": 30,
    "n_cities": 2,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,
    "seed": 0,
    "malfunction_rate": 1 / 200,
    "malfunction_min_duration": 20,
    "malfunction_max_duration": 50,
    "observation_max_path_depth": 30,
    "observation_tree_depth": 2,
}


def main(_: Any) -> None:
    """Run example."""

    # Environment.
    environment_factory = functools.partial(make_environment, **env_config)

    # Networks.
    network_factory = lp_utils.partial_kwargs(
        madqn.make_default_networks, architecture_type=ArchitectureType.recurrent
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
    program = madqn.MADQN(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=1,
        exploration_scheduler_fn=LinearExplorationScheduler(
            epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=1e-5
        ),
        batch_size=32,
        samples_per_insert=4,
        max_gradient_norm=20.0,
        min_replay_size=32,
        max_replay_size=5000,
        trainer_fn=madqn.MADQNRecurrentTrainer,
        executor_fn=madqn.MADQNRecurrentExecutor,
        checkpoint_subpath=checkpoint_dir,
        evaluator_interval={"executor_episodes": 2},
    ).build()

    # Ensure only trainer runs on gpu, while other processes run on cpu.
    local_resources = lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=["trainer"]
    )

    # Launch.
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
