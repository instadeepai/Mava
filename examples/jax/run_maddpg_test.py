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

# TODO(Arnu): remove mypy ignore when the system is running
# type: ignore

"""Test for running Jax MADDPG on debug MPE environment."""
from datetime import datetime

import optax

import mava.utils as utils
from mava.systems.tf import maddpg


def test_main() -> None:
    """Acceptance test for Jax MADDPG system."""

    # Create system and config
    system = maddpg.MADDPG()
    config = system.config

    # Environment.
    environment_factory = utils.make_factory(
        utils.environments.make_environment,
        env_name="debugging",
        action_space="continuous",
    )

    # Logger.
    logger_factory = utils.make_factory(
        utils.loggers.make_logger,
        directory="~/mava",
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=str(datetime.now()),
        time_delta=10,
    )

    # Networks.
    network_factory = utils.make_factory(maddpg.make_default_networks)

    # optimizers
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # update default config
    config.update(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        network_factory=network_factory,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
    )

    # Launch system
    system.launch(
        config=config, num_executors=2, nodes_on_gpu=["trainer"], name="maddpg"
    )
