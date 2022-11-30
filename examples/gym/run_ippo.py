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

"""Example running IPPO on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any
import gym
import optax
from absl import app, flags

from mava.components.component_groups import recurrent_policy_components
from mava.systems import ippo
from mava.wrappers.double_independent_gym import DoubleIndependentGym
from mava.utils.loggers import logger_utils
from mava.components.training import HuberValueLoss

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "CartPole-v1",
    "Debugging environment name (str).",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "logs", "Base dir to store experiments.")

def make_environment(env_name, evaluation=False):
    env = gym.make(env_name)
    env = DoubleIndependentGym(env)
    return env, f"Double {env_name}"

def main(_: Any) -> None:
    """Example running recurrent IPPO on debugging environment."""
    BATCH_SIZE = 256

    # Environment.
    environment_factory = functools.partial(
        make_environment,
        env_name=FLAGS.env_name,
    )

    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=[64, 64],
            critic_layer_sizes=[64, 64],
            policy_recurrent_layer_sizes=[64],
            policy_layers_after_recurrent=[64],
            critic_recurrent_layer_sizes=[64],
            critic_layers_after_recurrent=[64],
            orthogonal_initialisation=True,
            policy_network_head_weight_gain=0.01,
            critic_batch_size=BATCH_SIZE,
            layer_norm=True,
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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-3e-4)
    )

    critic_optimiser = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-3e-4)
    )

    system = ippo.IPPOSystem()
    system.update(recurrent_policy_components)
    system.update(HuberValueLoss)

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=experiment_path,
        policy_optimiser=policy_optimiser,
        critic_optimiser=critic_optimiser,
        run_evaluator=True,
        # sample_batch_size=320,
        epoch_batch_size=BATCH_SIZE,
        max_queue_size=BATCH_SIZE * 2,
        # max_queue_size=640,
        num_epochs=2,
        num_executors=4,
        multi_process=True,
        evaluation_interval={"executor_steps": 10000},
        evaluation_duration={"evaluator_episodes": 10},
        # evaluation_duration={"evaluator_episodes": 32},
        huber_delta=10.0,
        clip_value=True,
        clipping_epsilon=0.2,
        entropy_cost=3e-4,
        executor_parameter_update_period=20,
        normalize_advantage=False,
        normalize_target_values=False,
        num_minibatches=8,
        sequence_length=10,
        period=10,
        trainer_parameter_update_period=5,
        value_clip_parameter=0.2,
        value_cost=1.,
        max_gradient_norm = 0.5,
        normalize_observations=False,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
