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

"""MADDPG system implementation."""

from mava.components.building import distributor
import reverb

from mava.core import System
from mava.systems.building import Builder
from mava.components import building
from mava.components.tf import building as tf_building
from mava.utils import enums

# TODO (Arnu): figure out best way to send in system arguments
system_config = {"setup": {}, "table": {}, "dataset": {}}


class MADDPG(System):
    def __init__(self, config):

        self._config = config
        self._distribute = False

        # General setup
        setup = building.SystemSetup(
            network_sampling_setup=enums.NetworkSampler.fixed_agent_networks,
            trainer_networks=enums.Trainer.single_trainer,
            termination_condition=None,
        )

        # Replay table
        table = building.OffPolicyReplayTables(
            name=config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=config.max_replay_size,
            rate_limiter=building.OffPolicyRateLimiter(
                samples_per_insert=config.samples_per_insert,
                min_replay_size=config.min_replay_size,
            ),
            signature=building.ParallelSequenceAdderSignature(),
        )

        # Dataset
        dataset = building.DatasetIterator(
            batch_size=config.batch_size,
            prefetch_size=config.prefetch_size,
        )

        # Adder
        adder = building.ParallelNStepTransitionAdder(
            net_to_ints=config.net_to_ints,
            table_network_config=config.table_network_config,
            n_step=config.n_step,
            discount=config.discount,
        )

        # Variable server and clients
        variable_server = tf_building.VariableServer(
            checkpoint=config.checkpoint,
            checkpoint_subpath=config.checkpoint_subpath,
            checkpoint_minute_interval=config.checkpoint_minute_interval,
        )

        # Executor client
        executor_client = tf_building.ExecutorVariableClient(
            executor_variable_update_period=config.executor_variable_update_period
        )

        # Trainer client
        trainer_client = tf_building.TrainerVariableClient()

        # Executor
        executor = building.Executor()

        # Trainer
        trainer = building.Trainer()

        self._system_components = [
            setup,
            table,
            dataset,
            adder,
            variable_server,
            executor_client,
            trainer_client,
            executor,
            trainer,
        ]

    def build(self, name="maddpg"):
        self._name = name

        # Builder
        self._builder = Builder(components=self._system_components)
        self._builder.build()

    def distribute(self, num_executors=1, nodes_on_gpu=["trainer"]):
        self._distribute = True

        # Distributor
        distributor = building.Distributor(
            num_executors=num_executors,
            multi_process=True,
            nodes_on_gpu=nodes_on_gpu,
            name=self._name,
        )
        self._system_components.append(distributor)

    def launch(self):
        if not self._distribute:
            distributor = building.Distributor(multi_process=False)
            self._system_components.append(distributor)

        self._builder.launch()


## Example of create/launching system
system = MADDPG(system_config)

# Build system
system.build(name="maddpg")

# Distribute system processes
system.distribute(num_executors=2, nodes_on_gpu=["trainer"])

# Launch system
system.launch()