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

"""Tests for MADDPG."""

import functools

import launchpad as lp
import sonnet as snt

import mava
from mava.components.tf import architectures
from mava.components.tf.architectures.utils import fully_connected_network_spec
from mava.systems.tf import maddpg
from mava.utils import lp_utils
from mava.utils.enums import ArchitectureType
from mava.utils.environments import debugging_utils


class TestMADDPG:
    """Simple integration/smoke test for MADDPG."""

    def test_maddpg_on_debugging_env(self) -> None:
        """Test feedforward maddpg."""

        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="continuous",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(
            maddpg.make_default_networks, policy_networks_layer_sizes=(64, 64)
        )

        # system
        system = maddpg.MADDPG(
            environment_factory=environment_factory,
            network_factory=network_factory,
            num_executors=1,
            batch_size=32,
            min_replay_size=32,
            max_replay_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            checkpoint=False,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        # Launch gpu config - don't use gpu
        local_resources = lp_utils.to_device(
            program_nodes=program.groups.keys(), nodes_on_gpu=[]
        )

        lp.launch(
            program,
            launch_type="test_mt",
            local_resources=local_resources,
        )

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(2):
            trainer.step()

    def test_recurrent_maddpg_on_debugging_env(self) -> None:
        """Test recurrent maddpg."""
        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="continuous",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(
            maddpg.make_default_networks,
            archecture_type=ArchitectureType.recurrent,
            policy_networks_layer_sizes=(32, 32),
        )

        # system
        system = maddpg.MADDPG(
            environment_factory=environment_factory,
            network_factory=network_factory,
            num_executors=1,
            batch_size=16,
            min_replay_size=16,
            max_replay_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            checkpoint=False,
            trainer_fn=maddpg.training.MADDPGDecentralisedRecurrentTrainer,
            executor_fn=maddpg.execution.MADDPGRecurrentExecutor,
            sequence_length=4,
            period=4,
            bootstrap_n=2,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        # Launch gpu config - don't use gpu
        local_resources = lp_utils.to_device(
            program_nodes=program.groups.keys(), nodes_on_gpu=[]
        )

        lp.launch(
            program,
            launch_type="test_mt",
            local_resources=local_resources,
        )

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(2):
            trainer.step()

    def test_centralised_maddpg_on_debugging_env(self) -> None:
        """Test centralised maddpg."""
        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="continuous",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(
            maddpg.make_default_networks,
            policy_networks_layer_sizes=(32, 32),
        )

        # system
        system = maddpg.MADDPG(
            environment_factory=environment_factory,
            network_factory=network_factory,
            num_executors=1,
            batch_size=16,
            min_replay_size=16,
            max_replay_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            checkpoint=False,
            architecture=architectures.CentralisedQValueCritic,
            trainer_fn=maddpg.MADDPGCentralisedTrainer,
            shared_weights=False,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        # Launch gpu config - don't use gpu
        local_resources = lp_utils.to_device(
            program_nodes=program.groups.keys(), nodes_on_gpu=[]
        )

        lp.launch(
            program,
            launch_type="test_mt",
            local_resources=local_resources,
        )

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(2):
            trainer.step()

    def test_networked_maddpg_on_debugging_env(self) -> None:
        """Test networked maddpg."""
        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="continuous",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(
            maddpg.make_default_networks,
            policy_networks_layer_sizes=(32, 32),
        )

        # system
        system = maddpg.MADDPG(
            environment_factory=environment_factory,
            network_factory=network_factory,
            num_executors=1,
            batch_size=16,
            min_replay_size=16,
            max_replay_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            checkpoint=False,
            trainer_fn=maddpg.MADDPGNetworkedTrainer,
            architecture=architectures.NetworkedQValueCritic,
            connection_spec=fully_connected_network_spec,
            shared_weights=False,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        # Launch gpu config - don't use gpu
        local_resources = lp_utils.to_device(
            program_nodes=program.groups.keys(), nodes_on_gpu=[]
        )

        lp.launch(
            program,
            launch_type="test_mt",
            local_resources=local_resources,
        )

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(2):
            trainer.step()

    def test_state_based_maddpg_on_debugging_env(self) -> None:
        """Test state based maddpg."""
        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="continuous",
            return_state_info=True,
        )

        # networks
        network_factory = lp_utils.partial_kwargs(
            maddpg.make_default_networks,
            policy_networks_layer_sizes=(32, 32),
        )

        # system
        system = maddpg.MADDPG(
            environment_factory=environment_factory,
            network_factory=network_factory,
            num_executors=1,
            batch_size=16,
            min_replay_size=16,
            max_replay_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            checkpoint=False,
            trainer_fn=maddpg.MADDPGStateBasedTrainer,
            architecture=architectures.StateBasedQValueCritic,
            shared_weights=False,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        # Launch gpu config - don't use gpu
        local_resources = lp_utils.to_device(
            program_nodes=program.groups.keys(), nodes_on_gpu=[]
        )

        lp.launch(
            program,
            launch_type="test_mt",
            local_resources=local_resources,
        )

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(2):
            trainer.step()
