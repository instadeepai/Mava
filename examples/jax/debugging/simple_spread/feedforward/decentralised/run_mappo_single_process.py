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

import launchpad as lp
import optax
from absl import app

from mava.systems.jax import mappo
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils


def main(_: Any) -> None:
    """Main script for running system."""

    system = mappo.MAPPOSystem()

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(64, 64),
            *args,
            **kwargs,
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = str(datetime.now())
    checkpoint_subpath = f"{base_dir}/{mava_id}"
    # Log every [log_every] seconds.
    log_every = 1
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=mava_id,
        time_delta=log_every,
    )
    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0),
        optax.adam(1e-4),
    )
    # Build the system
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=10,
        multi_process=False,
        run_evaluator=True,
        num_executors=1,
        use_next_extras=False,
        max_queue_size=5000,
        sample_batch_size=2,
        max_in_flight_samples_per_worker=4,
        num_workers_per_iterator=-1,
        rate_limiter_timeout_ms=-1,
        checkpoint=True,
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING,
        sequence_length=4,
        period=4,
    )
    system.launch()


if __name__ == "__main__":
    app.run(main)
