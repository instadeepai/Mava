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

"""Example running MAMCTS on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any

import jax.numpy as jnp
import mctx
import optax
from absl import app, flags
from mctx import RecurrentFnOutput, RootFnOutput

from mava.systems.jax import mamcts
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

    def root_fn(params, key, observation):
        return RootFnOutput(
            prior_logits=jnp.zeros((1, 5)),
            value=jnp.array([1]),
            embedding=jnp.zeros((1, 5)),
        )

    def recurrent_fn(params, rng_key, action, observations):

        return (
            RecurrentFnOutput(
                reward=jnp.array([0]),
                discount=jnp.array([1]),
                prior_logits=jnp.zeros((1, 5)),
                value=jnp.array([1]),
            ),
            observations,
        )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mamcts.make_default_networks(
            root_fn=root_fn,
            recurrent_fn=recurrent_fn,
            search_algorithm=mctx.gumbel_muzero_policy,  # type: ignore
            policy_layer_sizes=(254, 254, 254),
            critic_layer_sizes=(512, 512, 256),
            *args,
            **kwargs,
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
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the system.
    system = mamcts.MAMCTSSystem()

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
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
