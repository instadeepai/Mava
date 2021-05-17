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

"""Tests for MADQN."""

import functools
from typing import Dict, Mapping, Sequence, Union

import launchpad as lp
import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import networks
from launchpad.nodes.python.local_multi_processing import PythonProcess

import mava
from mava import specs as mava_specs
from mava.components.tf.networks import epsilon_greedy_action_selector
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from mava.utils.loggers import Logger


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    epsilon: tf.Variable = tf.Variable(1.0, trainable=False),
    q_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256),
    shared_weights: bool = True,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(q_networks_layer_sizes, Sequence):
        q_networks_layer_sizes = {key: q_networks_layer_sizes for key in specs.keys()}

    def action_selector_fn(
        q_values: types.NestedTensor, legal_actions: types.NestedTensor
    ) -> types.NestedTensor:
        return epsilon_greedy_action_selector(
            action_values=q_values, legal_actions_mask=legal_actions
        )

    q_networks = {}
    action_selectors = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the policy network.
        q_network = snt.Sequential(
            [
                networks.LayerNormMLP(q_networks_layer_sizes[key], activate_final=True),
                networks.NearZeroInitializedLinear(num_dimensions),
            ]
        )

        # epsilon greedy action selector
        action_selector = action_selector_fn

        q_networks[key] = q_network
        action_selectors[key] = action_selector

    return {
        "q_networks": q_networks,
        "action_selectors": action_selectors,
    }


class TestMADQN:
    """Simple integration/smoke test for MADQN."""

    def test_madqn_on_debugging_env(self) -> None:
        """Tests that the system can run on the simple spread
        debugging environment without crashing."""

        # set loggers info
        # TODO Allow for no checkpointing and no loggers to be
        # passed in.
        mava_id = "tests/madqn"
        base_dir = "~/mava"
        log_info = (base_dir, f"{mava_id}/logs")

        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="discrete",
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

        system = madqn.MADQN(
            environment_factory=environment_factory,
            network_factory=network_factory,
            log_info=log_info,
            num_executors=2,
            batch_size=32,
            min_replay_size=32,
            max_replay_size=1000,
            optimizer=snt.optimizers.Adam(learning_rate=1e-3),
            checkpoint=False,
            checkpoint_subpath=checkpoint_dir,
            trainer_logger=trainer_logger,
            exec_logger=exec_logger,
            eval_logger=eval_logger,
        )

        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        # Launch gpu config - don't use gpu
        gpu_id = -1
        env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
        local_resources = {
            "trainer": PythonProcess(env=env_vars),
            "evaluator": PythonProcess(env=env_vars),
            "executor": PythonProcess(env=env_vars),
        }
        lp.launch(
            program,
            launch_type="test_mt",
            local_resources=local_resources,
        )

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(5):
            trainer.step()
