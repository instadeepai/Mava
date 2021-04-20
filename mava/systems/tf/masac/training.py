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


"""MASAC trainer implementation."""
import copy
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava

class BaseMASACTrainer(mava.Trainer):
    """MASAC trainer.
    This is the trainer component of a MASAC system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_Q_1_networks: Dict[str, snt.Module],
        critic_Q_2_networks: Dict[str, snt.Module],
        critic_V_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_V_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_V_optimizer: snt.Optimizer = None,
        critic_Q_1_optimizer: snt.Optimizer = None,
        critic_Q_2_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "Checkpoints",
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_V_network: the online critic for predicting state values.
          critic_Q_1_network: the online critic for predicting state-action values.
          critic_Q_2_network: the online critic for predicting state-action values.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_V_network: the target critic for predicting state values.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          dataset: dataset to learn from, whether fixed or from a replay buffer
            (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy and the critic.
          target_observation_network: the target observation network.
          policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
          critic_V_optimizer: the optimizer to be applied to the critic_V loss.
          critic_Q_1_optimizer: the optimizer to be applied to the critic_Q_1 loss.
          critic_Q_2_optimizer: the optimizer to be applied to the critic_Q_2 loss.
          clipping: whether to clip gradients by global norm.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._agents = agents
        self._agent_types = agent_types
        self._shared_weights = shared_weights

        # Store online and target networks.
        self._policy_networks = policy_networks
        self._critic_V_networks = critic__V_networks
        self._critic_Q_1_networks = critic_Q_1_networks
        self._critic_Q_2_networks = critic_Q_2_networks
        self._target_policy_networks = target_policy_networks
        self._target_critic_V_networks = target_critic_V_networks

        self._observation_networks = observation_networks
        self._target_observation_networks = target_observation_networks

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

        # Create optimizers if they aren't given.
        self._critic_V_optimizer = critic_V_optimizer or snt.optimizers.Adam(1e-4)
        self._critic_Q_1_optimizer = critic_Q_1_optimizer or snt.optimizers.Adam(1e-4)
        self._critic_Q_2_optimizer = critic_Q_2_optimizer or snt.optimizers.Adam(1e-4)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        # Expose the variables.
        policy_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "critic_V": {},
            "critic_Q_1": {},
            "critic_Q_2": {},
            "policy": {},
        }
        for agent_key in self.unique_net_keys:
            policy_network_to_expose = snt.Sequential(
                [
                    self._target_observation_networks[agent_key],
                    self._target_policy_networks[agent_key],
                ]
            )
            policy_networks_to_expose[agent_key] = policy_network_to_expose
            # TODO (dries): Determine why acme has a critic
            #  in self._system_network_variables
            self._system_network_variables["critic_V"][
                agent_key
            ] = target_critic_V_networks[agent_key].variables
            self._system_network_variables["critic_Q_1"][
                agent_key
            ] = critic_Q_1_networks[agent_key].variables
            self._system_network_variables["critic_Q_2"][
                agent_key
            ] = critic_Q_2_networks[agent_key].variables
            self._system_network_variables["policy"][
                agent_key
            ] = policy_network_to_expose.variables

        # Create checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            # TODO (dries): Address this new warning: WARNING:tensorflow:11 out
            #  of the last 11 calls to
            #  <function MultiDeviceSaver.save.<locals>.tf_function_save at
            #  0x7eff3c13dd30> triggered tf.function retracing. Tracing is
            #  expensive and the excessive number tracings could be due to (1)
            #  creating @tf.function repeatedly in a loop, (2) passing tensors
            #  with different shapes, (3) passing Python objects instead of tensors.
            for agent_key in self.unique_net_keys:
                objects_to_save = {
                    "counter": self._counter,
                    "policy": self._policy_networks[agent_key],
                    "critic_V": self._critic_V_networks[agent_key],
                    "critic_Q_1": self._critic_Q_1_networks[agent_key],
                    "critic_Q_2": self._critic_Q_2_networks[agent_key],
                    "observation": self._observation_networks[agent_key],
                    "target_policy": self._target_policy_networks[agent_key],
                    "target_critic_V": self._target_critic_V_networks[agent_key],
                    "target_observation": self._target_observation_networks[agent_key],
                    "policy_optimizer": self._policy_optimizer,
                    "critic_V_optimizer": self._critic_V_optimizer,
                    "critic_Q_1_optimizer": self._critic_Q_1_optimizer,
                    "critic__Q_2_optimizer": self._critic_Q_2_optimizer,
                    "num_steps": self._num_steps,
                }

                checkpointer_dir = os.path.join(checkpoint_subpath, agent_key)
                checkpointer = tf2_savers.Checkpointer(
                    time_delta_minutes=1,
                    add_uid=False,
                    directory=checkpointer_dir,
                    objects_to_save=objects_to_save,
                    enable_checkpointing=True,
                )
                self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None