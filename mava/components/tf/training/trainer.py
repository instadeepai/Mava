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

"""Trainer component"""

import copy
from typing import Any, Dict, Optional

import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.utils import loggers

from mava.callbacks import Callback
from mava.systems.training import SystemTrainer
from mava.systems.tf.variable_utils import VariableClient
from mava.utils import training_utils as train_utils
from mava.utils.sort_utils import sort_str_num

train_utils.set_growing_gpu_memory()


class Trainer(Callback):
    def __init__(
        self,
        config: Dict[str, Any],
        networks: Dict[str, Dict[str, snt.Module]],
        optimizers: Dict[str, Dict[str, snt.Optimizer]],
        dataset: tf.data.Dataset,
        variable_client: VariableClient,
    ) -> None:

        self.config = config
        self._agents = self.config["agents"]
        self._agent_net_keys = self.config["agent_net_keys"]
        self._variable_client = variable_client

        # Setup counts
        self._counts = self.config["counts"]

        # General learner book-keeping and loggers.
        self._logger = self.config["logger"] or loggers.make_default_logger("trainer")

        # Other learner parameters.
        self._discount = self.config["discount"]

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        # Store online and set up online networks.

        # Ensure obs and target networks are sonnet modules
        if "observation" in networks["online"].keys():
            self._observation_networks = {
                k: tf2_utils.to_sonnet_module(v)
                for k, v in networks["online"]["observation"].items()
            }

        # At minimum a system needs a policy network for each agent.
        self._policy_networks = networks["online"]["policy"]

        # Dictionary with unique network keys.
        self.unique_net_keys = sort_str_num(self._policy_networks.keys())

        if "critic" in networks["online"].keys():
            self._critic_networks = networks["online"]["critic"]

        # store and expose target networks
        if "target" in networks.keys():
            if "observation" in networks["target"].keys():
                self._target_observation_networks = {
                    k: tf2_utils.to_sonnet_module(v)
                    for k, v in networks["target"]["observation"].items()
                }

            if "policy" in networks["target"].keys():
                self._target_policy_networks = networks["target"]["policy"]

            if "target_critic" in networks["target"].keys():
                self._target_critic_networks = networks["target"]["critic"]

            # Expose target networks variables.
            policy_networks_to_expose = {}
            self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
                "policy": {},
            }
            if "critic" in networks["online"].keys():
                self._system_network_variables["critic"] = {}
            for agent_key in self.unique_net_keys:
                policy_network_to_expose = snt.Sequential(
                    [
                        self._target_observation_networks[agent_key],
                        self._target_policy_networks[agent_key],
                    ]
                )
                policy_networks_to_expose[agent_key] = policy_network_to_expose
                self._system_network_variables["critics"][agent_key] = networks[
                    agent_key
                ].variables
                self._system_network_variables["policies"][
                    agent_key
                ] = policy_network_to_expose.variables

        # # Set up gradient clipping.
        # if self.config["max_gradient_norm"] is not None:
        #     self._max_gradient_norm = tf.convert_to_tensor(
        #         self.config["max_gradient_norm"]
        #     )
        # else:  # A very large number. Infinity results in NaNs.
        #     self._max_gradient_norm = tf.convert_to_tensor(1e10)

        # Necessary to track when to update target networks.
        # self._num_steps = tf.Variable(0, dtype=tf.int32)
        # self._target_averaging = target_averaging
        # self._target_update_period = target_update_period
        # self._target_update_rate = target_update_rate

        # Create optimizers for different agent types.
        if not isinstance(optimizers["policy"], dict):
            self._policy_optimizers: Dict[str, snt.Optimizer] = {}
            for agent in self.unique_net_keys:
                self._policy_optimizers[agent] = copy.deepcopy(optimizers["policy"])
        else:
            self._policy_optimizers = optimizers["policy"]

        if "critic" in optimizers.keys():
            self._critic_optimizers: Dict[str, snt.Optimizer] = {}
            for agent in self.unique_net_keys:
                self._critic_optimizers[agent] = copy.deepcopy(optimizers["critic"])

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def on_training_init_start(self, trainer: SystemTrainer) -> None:

        trainer._policy_networks = self._policy_networks
        trainer._agent_net_keys = self._agent_net_keys
        trainer._adder = self._adder
        trainer._variable_client = self._variable_client