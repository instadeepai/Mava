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

"""Example running MADQN on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any

import optax
from absl import app, flags

from mava.systems.jax import madqn
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils
from mava.utils.schedules.linear_epsilon_scheduler import LinearEpsilonScheduler

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


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name=FLAGS.env_name,
        action_space=FLAGS.action_space,
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return madqn.make_default_networks(policy_layer_sizes=(254, 254, 254), **kwargs)

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    #checkpoint_subpath = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

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
    lr = 1e-3
    optimizer = optax.chain(
        optax.adam(learning_rate=lr),
    )

    # epsilon scheduler
    epsilon_scheduler = LinearEpsilonScheduler(1.0, 0.05, 2000)
    # Create the system.
    system = madqn.MADQNSystem()

    # Build the system.
    system.build(
        epsilon_scheduler=epsilon_scheduler,
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        #checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=10,
        multi_process=True,
        run_evaluator=True,
        num_executors=1,
        sample_batch_size=1,
        target_update_period=100,
        min_data_server_size=1_000,
        n_step=1,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
