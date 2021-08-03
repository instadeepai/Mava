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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dm_env
import numpy as np
import tensorflow as tf

from mava.systems.tf.mad4pg.execution import MAD4PGRecurrentExecutor
from mava.systems.tf.mad4pg.system import MAD4PG
from mava.systems.tf.maddpg.builder import MADDPGBuilder
from mava.systems.tf.maddpg.execution import MADDPGRecurrentExecutor
from mava.systems.tf.maddpg.system import MADDPG
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource

"""Population based training interface for multi-agent RL systems."""
supported_pbt_systems = [MADDPG, MAD4PG]
supported_pbt_executors = [MADDPGRecurrentExecutor, MAD4PGRecurrentExecutor]


# PBT variable source
class PBTVariableSource(MavaVariableSource):
    def __init__(
        self,
        variables: Dict[str, Any],
        checkpoint: bool,
        checkpoint_subpath: str,
        unique_net_keys: Sequence[str],
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
        self._gen_time_interval = 5 * 60
        self._last_checkpoint_time = time.time()
        self._last_gen_start_time = time.time()
        self._unique_net_keys = unique_net_keys

        # The mutate info. [minimum, maximum, mutation rate]
        self._mutate_info = {
            "discount": [0.0, 1.0, 0.01],
            "target_update_rate": [0.0, 1.0, 0.01],
            "target_update_period": [0, 1000, 5],
        }

        super().__init__(
            variables=variables,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

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

    def can_update_vars(self, var_names: Sequence[str], worked_id: str) -> bool:
        """Check if the variables can be updated.
        Args:
            var_names (List[str]): Names of the variables to check.
            worked_id (str): The id of the worker that is making the request.
        Returns:
            can_update (bool): True if the variables can be updated.
        """
        return (
            self.variables["worker_gen_version"][worked_id]
            == self.variables["gen_version"]
        )

    def copy_and_mutate(self, weak_net_key: str, strong_net_key: str) -> None:
        """
        Copy and mutate the variables from the stronger network
        to the weaker network.
        Args:
            weak_net_key (str): The id of the network that should be mutated.
            strong_net_key (str): The id of the network that should be copied.
        Returns:
            None
        """

        # Copy the networks
        for net_type in [
            "policies",
            "observations",
            "target_policies",
            "target_observations",
            "critics",
            "target_critics",
        ]:
            weak_key = f"{weak_net_key}_{net_type}"
            strong_key = f"{strong_net_key}_{net_type}"
            # Loop through tuple
            # TODO (dries): Should we add some mutation here as well?
            for var_i in range(len(self.variables[strong_key])):
                self.variables[weak_key][var_i].assign(
                    self.variables[strong_key][var_i]
                )

        # Copy and mutate the hyperparameters (within bounding boxes)
        # TODO (dries): Add this back in again.
        for net_type in ["discount", "target_update_rate", "target_update_period"]:
            weak_key = f"{weak_net_key}_{net_type}"
            strong_key = f"{strong_net_key}_{net_type}"
            original_val = self.variables[strong_key]

            m_min, m_max, m_rate = self._mutate_info[net_type]  # type: ignore

            if original_val.dtype == tf.int32:
                mutation_val = np.random.randint(-m_rate, m_rate)  # type: ignore
            else:
                mutation_val = np.random.uniform(-m_rate, m_rate)  # type: ignore
            self.variables[weak_key].assign(
                np.clip(original_val + mutation_val, m_min, m_max)  # type: ignore
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

                # Get the rewards and reset them
                net_list = []
                reward_list = []
                for net_key in self._unique_net_keys:
                    net_list.append(net_key)
                    net_reward = self.variables[f"{net_key}_moving_avg_rewards"]
                    reward_list.append(net_reward)
                sorted_ind = np.argsort(reward_list)
                sorted_nets = np.array(net_list)[sorted_ind]

                # For the lowest 20% sample randomly from the top 20%.
                # After sampling add some random noise.
                lowest_20_percent = int(len(sorted_nets) * 0.2 + 1)
                for i in range(lowest_20_percent):
                    weak_net = sorted_nets[i]
                    strong_net = sorted_nets[
                        np.random.randint(lowest_20_percent, len(sorted_nets))
                    ]
                    self.copy_and_mutate(weak_net, strong_net)

                # Reset the rewards
                for net_key in self._unique_net_keys:
                    self.variables[f"{net_key}_moving_avg_rewards"].assign(0.0)

                # Increament the generation
                self.variables["gen_version"].assign_add(1)


def BasePBTWrapper(  # noqa
    system: Union[MADDPG, MAD4PG],  # noqa
) -> Union[MADDPG, MAD4PG]:
    """Initializes the broadcaster communicator.
    Args:
        system: The system that should be wrapped.
    Returns:
        system: The wrapped system.
    """
    if type(system) not in supported_pbt_systems:
        raise NotImplementedError(
            f"Currently only the {supported_pbt_systems} systems have "
            f"the correct hooks to support PBT. Not {type(system)}."
        )

    if system._builder._executor_fn not in supported_pbt_executors:
        raise NotImplementedError(
            f"Currently only the {supported_pbt_executors} executors have "
            f"the correct hooks to support PBT. Not {system._builder._executor_fn}."
        )

    # Wrap the executor with the PBT hooks
    class PBTExecutor(system._builder._executor_fn):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)

        # PBT executor
        def _custom_end_of_episode_logic(
            self, timestep: dm_env.TimeStep, agent_net_keys: Dict[str, str]
        ) -> None:
            """Custom logic at the end of an episode."""
            mvg_avg_weighting = 0.01
            rewards = timestep.reward
            for agent in rewards.keys():
                net_key = self._agent_net_keys[agent]
                self._variable_client.move_avg_and_wait(
                    f"{net_key}_moving_avg_rewards", rewards[agent], mvg_avg_weighting
                )

    system._builder._executor_fn = PBTExecutor

    # Wrap the system builder with the PBT hooks
    class PBTBuilder(type(system._builder)):  # type: ignore
        def __init__(
            self,
            builder: MADDPGBuilder,
        ):
            """Initialise the system.
            Args:
                builder: The builder to wrap.
            """

            self.__dict__ = builder.__dict__

        def create_custom_var_server_variables(
            self,
            variables: Dict[str, tf.Variable],
        ) -> None:
            """Create counter variables.
            Args:
                variables (Dict[str, snt.Variable]): dictionary with
                variable_source variables in.
            Returns:
                variables (Dict[str, snt.Variable]): dictionary with
                variable_source variables in.
            """
            var_server_vars = {
                "gen_version": tf.Variable(0, dtype=tf.int32),
                "worker_gen_version": {
                    trainer_id: tf.Variable(0, dtype=tf.int32)
                    for trainer_id in self._config.trainer_networks.keys()
                },
            }
            variables.update(var_server_vars)

        def create_custom_trainer_variables(
            self,
            variables: Dict[str, tf.Variable],
            get_keys: List[str] = None,
        ) -> None:
            """Create counter variables.
            Args:
                variables (Dict[str, snt.Variable]): dictionary with variable_source
                variables in.
                get_keys (List[str]): list of keys to get from the variable server.
            Returns:
                None.
            """
            hyper_vars = {}
            for net_key in self._config.unique_net_keys:
                hyper_vars[f"{net_key}_discount"] = tf.Variable(0.99, dtype=tf.float32)
                hyper_vars[f"{net_key}_target_update_rate"] = tf.Variable(
                    0.01, dtype=tf.float32
                )
                hyper_vars[f"{net_key}_target_update_period"] = tf.Variable(
                    100, dtype=tf.int32
                )
            if get_keys:
                get_keys.extend(hyper_vars)
            variables.update(hyper_vars)

        def create_custom_executor_variables(
            self,
            variables: Dict[str, tf.Variable],
            get_keys: List[str] = None,
        ) -> None:
            """Create counter variables.
            Args:
                variables (Dict[str, snt.Variable]): dictionary with variable_source
                variables in.
                get_keys (List[str]): list of keys to get from the variable server.
            Returns:
                None.
            """
            reward_vars = {}
            for net_key in self._config.unique_net_keys:
                reward_vars[f"{net_key}_moving_avg_rewards"] = tf.Variable(
                    0, dtype=tf.float32
                )
            if get_keys:
                get_keys.extend(reward_vars)
            variables.update(reward_vars)

        def variable_server_fn(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> PBTVariableSource:
            """Create a variable server.
            Args:
                *args: Variable arguments.
                **kwargs: Variable keyword arguments.
            Returns:
                a PBTVariableSource.
            """
            return PBTVariableSource(*args, **kwargs)

        def get_hyper_parameters(
            self,
            discount: float,
            target_update_rate: Optional[float],
            target_update_period: Optional[int],
            variables: Dict[str, tf.Variable],
        ) -> Tuple[tf.Variable, tf.Variable, tf.Variable]:
            """Get the hyperparameters.
            Args:
                discount (float): the discount factor
                target_update_rate (float): the rate at which the target network is
                updated.
                target_update_period (int): the period at which the target network is
                updated.
                variables (Dict[str, tf.Variable]): the variables in the system.
            Returns:
                discounts (Dict[str, tf.Variable]): the discount factors
                target_update_rate (Dict[str, tf.Variable]): the rates at which the
                target network is updated.
                target_update_period (Dict[str, tf.Variable]): the periods at which the
                target network is updated.
            """
            discounts = {
                net_key: variables[f"{net_key}_discount"]
                for net_key in self._config.unique_net_keys
            }
            target_update_rates = {
                net_key: variables[f"{net_key}_target_update_rate"]
                for net_key in self._config.unique_net_keys
            }
            target_update_periods = {
                net_key: variables[f"{net_key}_target_update_period"]
                for net_key in self._config.unique_net_keys
            }
            return discounts, target_update_rates, target_update_periods

    system._builder = PBTBuilder(system._builder)
    return system
