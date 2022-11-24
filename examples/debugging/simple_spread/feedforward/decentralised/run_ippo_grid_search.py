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

"""Example running IPPO on debug MPE environments, using grid search over num_epochs."""
import functools
from datetime import datetime
from typing import Any, Dict

import optax
from absl import app, flags

from mava.systems import ippo
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

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


def create_and_run_lp_program(config: Dict) -> None:
    """Code to run lp program using config.

    Args:
        config : hyperparam config.
    """
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name=FLAGS.env_name,
        action_space=FLAGS.action_space,
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=(64, 64),
            critic_layer_sizes=(64, 64, 64),
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
        epoch_batch_size=5,
        num_epochs=config["num_epochs"],
        num_executors=1,
        multi_process=True,
        clip_value=False,
        termination_condition={"executor_steps": 10000},
        # Wait for worker to finish - this critical to waiting for program to finish.
        wait=True,
    )
    # Launch the system.
    system.launch()


def main(_: Any) -> None:
    """Runs lp inside a new thread.

    Args:
        _ : unused.
    """
    config = {}
    # Loop through a hyperparam such as num_epochs
    for num_epochs in [5, 10, 15]:
        config["num_epochs"] = num_epochs
        create_and_run_lp_program(config)
        print(f"Completed: {config}")


if __name__ == "__main__":
    app.run(main)
