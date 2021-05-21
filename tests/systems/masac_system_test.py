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

"""Tests for MASAC."""

import functools
from typing import Any, Dict, Mapping, Sequence, Union

import launchpad as lp
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils

import mava
from mava import specs as mava_specs
from mava.systems.tf import masac
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from mava.utils.loggers import Logger


class ActorNetwork(snt.Module):
    def __init__(
        self,
        n_hidden_unit1: int,
        n_hidden_unit2: int,
        n_actions: int,
        logprob_epsilon: float,
        observation_netork: snt.Module,
    ):
        super(ActorNetwork, self).__init__()
        self.logprob_epsilon = tf.Variable(logprob_epsilon)
        self.observation_network = observation_netork
        w_bound = tf.Variable(3e-3)
        self.hidden1 = snt.Linear(n_hidden_unit1)
        self.hidden2 = snt.Linear(n_hidden_unit2)

        self.mean = snt.Linear(
            n_actions,
            w_init=snt.initializers.RandomUniform(-w_bound, w_bound),
            b_init=snt.initializers.RandomUniform(-w_bound, w_bound),
        )
        self.log_std = snt.Linear(
            n_actions,
            w_init=snt.initializers.RandomUniform(-w_bound, w_bound),
            b_init=snt.initializers.RandomUniform(-w_bound, w_bound),
        )

    def __call__(self, x: Any) -> Any:
        """forward call for sonnet module"""
        x = self.observation_network(x)
        x = self.hidden1(x)
        x = tf.nn.relu(x)
        x = self.hidden2(x)
        x = tf.nn.relu(x)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - tf.math.log(
            1.0 - tf.pow(squashed_actions, 2) + self.logprob_epsilon
        )
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)
        return squashed_actions, logprob


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_networks_V_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        512,
        512,
        256,
    ),
    critic_networks_Q_1_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        512,
        512,
        256,
    ),
    critic_networks_Q_2_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        512,
        512,
        256,
    ),
    shared_weights: bool = True,
    sigma: float = 0.3,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_V_layer_sizes, Sequence):
        critic_networks_V_layer_sizes = {
            key: critic_networks_V_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_Q_1_layer_sizes, Sequence):
        critic_networks_Q_1_layer_sizes = {
            key: critic_networks_Q_1_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_Q_2_layer_sizes, Sequence):
        critic_networks_Q_2_layer_sizes = {
            key: critic_networks_Q_2_layer_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    critic_V_networks = {}
    critic_Q_1_networks = {}
    critic_Q_2_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(specs[key].actions.shape, dtype=int)

        # Create the shared observation network; here simply a state-less operation.
        observation_network = tf2_utils.to_sonnet_module(tf2_utils.batch_concat)

        # Create the policy network.
        policy_network = ActorNetwork(
            256, 256, num_dimensions, 0.3, observation_network
        )

        # Create the critic network.
        critic_V_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks.LayerNormMLP(
                    critic_networks_V_layer_sizes[key], activate_final=False
                ),
                snt.Linear(1),
            ]
        )

        # Create the critic network.
        critic_Q_1_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(
                    critic_networks_Q_1_layer_sizes[key], activate_final=False
                ),
                snt.Linear(1),
            ]
        )

        # Create the critic network.
        critic_Q_2_network = snt.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(
                    critic_networks_Q_2_layer_sizes[key], activate_final=False
                ),
                snt.Linear(1),
            ]
        )
        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_V_networks[key] = critic_V_network
        critic_Q_1_networks[key] = critic_Q_1_network
        critic_Q_2_networks[key] = critic_Q_2_network

    return {
        "policies": policy_networks,
        "critics_V": critic_V_networks,
        "critics_Q_1": critic_Q_1_networks,
        "critics_Q_2": critic_Q_2_networks,
        "observations": observation_networks,
    }


class TestMASAC:
    """Simple integration/smoke test for MASAC."""

    def test_masac_on_debugging_env(self) -> None:
        """Tests that the system can run on the simple spread
        debugging environment without crashing."""

        # set loggers info
        # TODO Allow for no checkpointing and no loggers to be
        # passed in.
        mava_id = "tests/maddpg"
        base_dir = "~/mava"
        log_info = (base_dir, f"{mava_id}/logs")

        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="continuous",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(make_networks)

        # system
        checkpoint_dir = f"{base_dir}/{mava_id}"

        log_every = 10
        trainer_logger = Logger(
            label="system_trainer",
            directory=base_dir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=mava_id,
            time_delta=log_every,
        )

        exec_logger = Logger(
            # _{executor_id} gets appended to label in system.
            label="train_loop_executor",
            directory=base_dir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=mava_id,
            time_delta=log_every,
        )

        eval_logger = Logger(
            label="eval_loop",
            directory=base_dir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=mava_id,
            time_delta=log_every,
        )

        # system
        system = masac.MASAC(
            environment_factory=environment_factory,
            network_factory=network_factory,
            log_info=log_info,
            num_executors=2,
            batch_size=32,
            min_replay_size=32,
            max_replay_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_V_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_Q_1_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_Q_2_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            checkpoint=False,
            checkpoint_subpath=checkpoint_dir,
            trainer_logger=trainer_logger,
            exec_logger=exec_logger,
            eval_logger=eval_logger,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        lp.launch(program, launch_type="test_mt")

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(5):
            trainer.step()
