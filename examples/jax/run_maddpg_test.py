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

"""Test for running Jax MADDPG on debug MPE environment."""
from datetime import datetime

import optax
from mava.systems.tf import maddpg
import mava.utils as utils


def test_main() -> None:

    config = maddpg.config

    # Environment.
    config.environment_factory = utils.make_factory(
        utils.environments.make_environment,
        env_name="debugging",
        action_space="continuous",
    )

    # Logger.
    config.logger_factory = utils.make_factory(
        utils.loggers.make_logger,
        directory="~/mava",
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=str(datetime.now()),
        time_delta=10,
    )

    # Networks.
    config.network_factory = utils.make_factory(maddpg.make_default_networks)

    # optimizers
    config.optimizer.policy = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )
    config.optimizer.critic = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Build system
    system = maddpg.MADDPG(config)

    # Launch system
    system.launch(num_executors=1, nodes_on_gpu=["trainer"], name="maddpg")
