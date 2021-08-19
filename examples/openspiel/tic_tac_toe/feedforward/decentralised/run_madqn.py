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

"""Example running MADQN on OpenSpiel's tic_tac_toe."""
import functools
from datetime import datetime
from typing import Any

import dm_env
import launchpad as lp
import sonnet as snt
from absl import app, flags
from open_spiel.python import rl_environment  # type: ignore

from mava.components.tf.modules.exploration import LinearExplorationScheduler
from mava.environment_loops.open_spiel_environment_loop import (
    OpenSpielSequentialEnvironmentLoop,
)
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.loggers import logger_utils
from mava.wrappers.open_spiel import OpenSpielSequentialWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "tic_tac_toe", "Name of the game")
flags.DEFINE_integer("num_players", None, "Number of players")

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def make_environment(
    evaluation: bool = False, game: str = FLAGS.game
) -> dm_env.Environment:
    raw_environment = rl_environment.Environment(game)
    environment = OpenSpielSequentialWrapper(raw_environment)
    return environment


def main(_: Any) -> None:

    # environment
    environment_factory = make_environment

    # Networks.
    network_factory = lp_utils.partial_kwargs(madqn.make_default_networks)

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
        epsilon_decay=1e-4,
        optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        checkpoint_subpath=checkpoint_dir,
        train_loop_fn=OpenSpielSequentialEnvironmentLoop,
        eval_loop_fn=OpenSpielSequentialEnvironmentLoop,
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
