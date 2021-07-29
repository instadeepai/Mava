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

import time
from typing import Any, Dict, Sequence, Union

import tensorflow as tf

from mava.systems.tf.mad4pg.system import MAD4PG
from mava.systems.tf.maddpg.builder import MADDPGBuilder
from mava.systems.tf.maddpg.system import MADDPG
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource

"""Base communication interface for multi-agent RL systems"""
supported_pbt_systems = [MADDPG, MAD4PG]


class PBTBuilder:
    def __init__(
        self,
        builder: MADDPGBuilder,
    ):
        """Initialise the system.
        Args:
            config (MADDPGConfig): system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn (Union[ Type[training.MADDPGBaseTrainer],
                Type[training.MADDPGBaseRecurrentTrainer], ], optional): Trainer
                function, of a correpsonding type to work with the selected system
                architecture. Defaults to training.MADDPGDecentralisedTrainer.
            executor_fn (Type[core.Executor], optional): Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to MADDPGFeedForwardExecutor.
            extra_specs (Dict[str, Any], optional): defines the specifications of extra
                information used by the system. Defaults to {}.
        """
        self._builder = builder

        # Overwrite methods
        # TODO (dries): Is there a better methode of doing this?
        # Basically I just want to overwrite some of the functions in MADDPGBuilder
        self._builder.create_hyperparameter_variables = (
            self.create_hyperparameter_variables
        )
        self._builder.variable_server_fn = self.variable_server_fn
        self._builder.get_hyper_parameters = self.get_hyper_parameters

    def create_hyperparameter_variables(
        self, variables: Dict[str, tf.Variable]
    ) -> Dict[str, tf.Variable]:
        """Create counter variables.
        Args:
            variables (Dict[str, snt.Variable]): dictionary with variable_source
            variables in.
        Returns:
            variables (Dict[str, snt.Variable]): dictionary with variable_source
            variables in.
        """
        hyper_vars = {}
        for net_key in self._builder._config.unique_net_keys:
            hyper_vars[f"{net_key}_discount"] = tf.Variable(0, dtype=tf.float32)
            hyper_vars[f"{net_key}_target_update_rate"] = tf.Variable(
                0, dtype=tf.float32
            )
            hyper_vars[f"{net_key}_target_update_period"] = tf.Variable(
                0, dtype=tf.int32
            )
        variables.update(hyper_vars)
        return hyper_vars.keys()

    def variable_server_fn(
        self, variables, checkpoint, checkpoint_subpath, unique_net_keys, trainer_ids
    ):
        return PBTVariableSource(
            variables, checkpoint, checkpoint_subpath, unique_net_keys, trainer_ids
        )

    def get_hyper_parameters(
        self, discount, target_update_rate, target_update_period, variables, get_keys
    ):
        """Get the hyperparameters.
        Args:
            discount (float): the discount factor
            target_update_rate (float): the rate at which the target network is
            target_update_period (int): the period at which the target network is
            variables (Dict[str, tf.Variable]): the variables in the system.
            get_keys (List[str]): the keys of the variables to get.
        Returns:
            hyper_parameters (Dict[str, tf.Variable]): the hyperparameters.
        """
        hyper_names = self.create_hyperparameter_variables(variables)
        get_keys.extend(hyper_names)
        discounts = {
            net_key: variables[f"{net_key}_discount"]
            for net_key in self._builder._config.unique_net_keys
        }
        target_update_rates = {
            net_key: variables[f"{net_key}_target_update_rate"]
            for net_key in self._builder._config.unique_net_keys
        }
        target_update_periods = {
            net_key: variables[f"{net_key}_target_update_period"]
            for net_key in self._builder._config.unique_net_keys
        }
        return discounts, target_update_rates, target_update_periods

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._builder, name)


class PBTVariableSource(MavaVariableSource):
    def __init__(
        self,
        variables: Dict[str, Any],
        checkpoint: bool,
        checkpoint_subpath: str,
        unique_net_keys: Sequence[str],
        trainer_ids: Sequence[str],
    ) -> None:
        """Initialise the variable source
        Args:
            variables (Dict[str, Any]): a dictionary with
            variables which should be stored in it.
            checkpoint (bool): Indicates whether checkpointing should be performed.
            checkpoint_subpath (str): checkpoint path
        Returns:
            None
        """
        # TODO (dries): Change this back to 5 * 60 for self._checkpoint_interval
        self._checkpoint_interval = 5 * 60
        self._gen_time_interval = 10 * 60
        self._last_checkpoint_time = time.time()
        self._last_gen_start_time = time.time()
        self._unique_net_keys = unique_net_keys
        self._trainer_ids = trainer_ids

        super().__init__(
            variables=variables,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    def get_init_custom_variables(self) -> None:
        """Initialise custom variables.
        Args:
            variables (Dict[str, Any]): The custom variables to initialise.
        Returns:
            None
        """
        custom_vars = {
            "gen_version": tf.Variable(0, dtype=tf.int32),
            "worker_gen_version": {
                trainer_id: tf.Variable(0, dtype=tf.int32)
                for trainer_id in self._trainer_ids
            },
        }

        # TODO: Fix this
        for net_key in self._unique_net_keys:
            custom_vars[f"{net_key}_discount"] = tf.Variable(0.99, tf.float32)
            custom_vars[f"{net_key}_target_update_rate"] = tf.Variable(0.01, tf.float32)
            custom_vars[f"{net_key}_target_update_period"] = tf.Variable(100, tf.int32)
        return custom_vars

    def custom_get_logic(self, var_names: Sequence[str], worked_id: str) -> None:
        """Custom logic to get variables.
        Args:
            var_names (Sequence[str]): Names of the variables to get.
            worked_id (str): The id of the worker that is currently working.
        Returns:
            None
        """
        # Set the generation version of the trainer to the latest generation version
        if worked_id.split("_")[0] == "trainer":
            self.variables["worker_gen_version"][worked_id].assign(
                self.variables["gen_version"]
            )

    def can_update_vars(self, var_names, worked_id) -> bool:
        """Check if the variables can be updated.
        Args:
            var_names (Sequence[str]): Names of the variables to update.
            worked_id (str): The id of the worker that called the set function.
        Returns:
            can_update (bool): True if the variables can be updated.
        """
        return (
            self.variables["worker_gen_version"][worked_id]
            == self.variables["gen_version"]
        )

    def run(self) -> None:
        """Run the variable source. This function allows for
        checkpointing and other centralised computations to
        be performed by the variable source.
                Args:
                    None
                Returns:
                    None
        """
        # Checkpoints every 5 minutes
        while True:
            # Wait 10 seconds before checking again
            time.sleep(10)

            # Check if system should checkpoint (+1 second to make sure the
            # checkpointer does the save)
            if (
                self._system_checkpointer
                and self._last_checkpoint_time + self._checkpoint_interval + 1
                < time.time()
            ):
                self._last_checkpoint_time = time.time()
                self._system_checkpointer.save()
                tf.print("Saved latest variables.")

            # Increment generation if necesary
            if self._last_gen_start_time + self._gen_time_interval < time.time():
                self._last_gen_start_time = time.time()
                tf.print("Starting a new generation.")
                # self.variables["gen_version"] += 1


class BasePBTModule:
    """Base class for PBT using a MARL system.
    Objects which implement this interface provide a set of functions
    to create systems that can perform some form of communication between
    agents in a multi-agent RL system.
    """

    def __init__(
        self,
        system: Union[MADDPG, MAD4PG],
    ) -> None:
        """Initializes the broadcaster communicator.
        Args:
            architecture: the BaseArchitecture used.
            shared: if a shared communication channel is used.
            channel_noise: stddev of normal noise in channel.
        """
        if type(system) not in supported_pbt_systems:
            raise NotImplementedError(
                f"Currently only {supported_pbt_systems} has "
                "the correct hooks to support PBT."
            )

        self._system = system
        # Wrap the system builder the PBT hooks
        self._system._builder = PBTBuilder(self._system._builder)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._system, name)
