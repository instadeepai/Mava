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
from typing import Any, Dict

import launchpad as lp
import sonnet as snt
from absl import app, flags
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava.components.tf.modules.exploration.exploration_scheduling import (
    ExponentialExplorationScheduler,
)
from mava.components.tf.modules.stabilising.fingerprints import FingerPrintStabalisation
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.environments.flatland_utils import flatland_env_factory
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "logs", "Base dir to store experiments.")


# flatland environment config
rail_gen_cfg: Dict = {
    "max_num_cities": 4,
    "max_rails_between_cities": 2,
    "max_rails_in_city": 3,
    "grid_mode": False,
    "seed": 0,
}

flatland_env_config: Dict = {
    "number_of_agents": 5,
    "width": 25,
    "height": 25,
    "rail_generator": sparse_rail_generator(**rail_gen_cfg),
    "schedule_generator": sparse_schedule_generator(),
    "obs_builder_object": TreeObsForRailEnv(
        max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=30)
    ),
}


def main(_: Any) -> None:

    # Environment.
    environment_factory = functools.partial(
        flatland_env_factory, env_config=flatland_env_config, include_agent_info=False
    )

    # Networks.
    network_factory = lp_utils.partial_kwargs(
        madqn.make_default_networks, policy_networks_layer_sizes=(256, 256)
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
        num_executors=4,
        exploration_scheduler=ExponentialExplorationScheduler,
        # fingerprint_fn=FingerPrintStabalisation,
        epsilon_min=0.01,
        epsilon_decay=0.999994,
        max_replay_size=500000,
        # importance_sampling_exponent=0.2,
        optimizer=snt.optimizers.Adam(learning_rate=5e-4),
        checkpoint_subpath=checkpoint_dir,
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
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
