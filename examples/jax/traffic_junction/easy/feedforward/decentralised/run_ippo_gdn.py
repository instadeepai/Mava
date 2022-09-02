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

"""Example running IPPO with GDNs on Easy Traffic Junction environment.

This should be able to solve the environment.
"""
import functools
from datetime import datetime
from typing import Any

import optax
from absl import app, flags

from mava.components.jax.communication.forward_pass import FeedforwardExecutorGdn
from mava.components.jax.communication.gdn_networks import (
    DefaultGdnNetworks,
    make_default_gcn,
)
from mava.components.jax.communication.graph_construction import GdnGraphFromEnvironment
from mava.systems.jax import ippo
from mava.utils.environments import traffic_junction_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "difficulty",
    "easy",
    "Traffic Junction Environment difficulty ('easy' | 'medium' | 'hard')",
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
        _ : Do not use args.
    """
    # Environment.
    environment_factory = functools.partial(
        traffic_junction_utils.make_environment,
        difficulty=FLAGS.difficulty,
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=(254, 254, 254),
            critic_layer_sizes=(512, 512, 256),
            single_network=False,
            *args,
            **kwargs,
        )

    # GDN network
    def gdn_network_factory(*args: Any, **kwargs: Any) -> Any:
        return make_default_gcn(  # type: ignore
            update_node_layer_sizes=((128, 128), (128, 128)), *args, **kwargs
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_subpath = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

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
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the system.
    system = ippo.IPPOSystemSeparateNetworks()

    # Manually add GDN components
    # TODO(Matthew): make a separate system for this later that builds on IPPO
    system.add(GdnGraphFromEnvironment)
    system.add(FeedforwardExecutorGdn)
    system.add(DefaultGdnNetworks)

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        gdn_network_factory=gdn_network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
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
