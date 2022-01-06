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


import functools
from datetime import datetime
from typing import Any

import launchpad as lp
import sonnet as snt
from absl import app, flags

from mava.components.tf.modules.exploration import LinearExplorationTimestepScheduler
from mava.systems.tf import vdn
from mava.utils import lp_utils
from mava.utils.environments import pettingzoo_utils
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


def main(_: Any) -> None:
    """Example running VDN on multi-agent Starcraft 2 (SMAC) environment."""
    # environment
    environment_factory = functools.partial(
        pettingzoo_utils.make_environment, env_class="smac", env_name=FLAGS.map_name
    )

    # Networks.
    network_factory = lp_utils.partial_kwargs(
        vdn.make_default_networks, policy_networks_layer_sizes=[64, 64]
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
    program = vdn.VDN(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=1,
        exploration_scheduler_fn=LinearExplorationTimestepScheduler(
            epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_steps=50000
        ),
        optimizer=snt.optimizers.RMSProp(
            learning_rate=0.0005, epsilon=0.00001, decay=0.99
        ),
        checkpoint_subpath=checkpoint_dir,
        batch_size=512,
        executor_variable_update_period=100,
        target_update_period=200,
        max_gradient_norm=10.0,
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
