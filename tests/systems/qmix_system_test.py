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


import functools

import launchpad as lp
import sonnet as snt

import mava
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf import value_decomposition
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils

"""Test for QMIX System"""


class TestQMIX:
    """Simple integration test for QMIX on Simple Spread enviromnent"""

    def test_qmix_on_debug_simple_spread(self) -> None:
        """Test recurrent QMIX."""

        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="discrete",
            return_state_info=True,
        )

        # Networks.
        network_factory = lp_utils.partial_kwargs(
            value_decomposition.make_default_networks,
        )

        # system
        system = value_decomposition.ValueDecomposition(
            environment_factory=environment_factory,
            network_factory=network_factory,
            mixer="qmix",
            num_executors=1,
            exploration_scheduler_fn=LinearExplorationScheduler(
                epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=1e-5
            ),
            optimizer=snt.optimizers.RMSProp(
                learning_rate=0.0005, epsilon=0.00001, decay=0.99
            ),
            batch_size=1,
            executor_variable_update_period=200,
            target_update_period=200,
            max_gradient_norm=20.0,
            min_replay_size=1,
            max_replay_size=10000,
            samples_per_insert=None,
            evaluator_interval={"executor_episodes": 2},
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
