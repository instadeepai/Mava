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

"""Example running MAPPO on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any

import numpy as np
import optax
from absl import app, flags
from acme.jax.networks.atari import DeepAtariTorso
from supersuit import dtype_v0

from mava.systems.jax import mappo
from mava.utils.environments import pettingzoo_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_class",
    "butterfly",
    "Pettingzoo environment class, e.g. atari (str).",
)
flags.DEFINE_string(
    "env_name",
    "cooperative_pong_v5",
    "Pettingzoo environment name, e.g. pong (str).",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)

flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Environment.
    environment_factory = functools.partial(
        pettingzoo_utils.make_environment,
        env_class=FLAGS.env_class,
        env_name=FLAGS.env_name,
        env_preprocess_wrappers=[(dtype_v0, {"dtype": np.float32})],
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        obs_net_forward = lambda x: DeepAtariTorso()(x)  # noqa: E731
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(64,),
            critic_layer_sizes=(256,),
            observation_network=obs_net_forward,
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

    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the system.
    system = mappo.MAPPOSystem()

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=experiment_path,
        optimizer=optimizer,
        run_evaluator=True,
        sample_batch_size=5,
        num_epochs=15,
        num_executors=1,
        multi_process=True,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
