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

"""Tests for MAPPO."""

import functools

import launchpad as lp
import sonnet as snt

import mava
from mava.systems.tf import mappo
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils


class TestMAPPO:
    """Simple integration/smoke test for MAPPO."""

    def test_mappo_on_debugging_env(self) -> None:
        """Test feedforward mappo."""
        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="discrete",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(
            mappo.make_default_networks, policy_networks_layer_sizes=(64, 64)
        )

        # system
        system = mappo.MAPPO(
            environment_factory=environment_factory,
            network_factory=network_factory,
            num_executors=1,
            batch_size=32,
            max_queue_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-3),
            critic_optimizer=snt.optimizers.Adam(learning_rate=1e-3),
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
