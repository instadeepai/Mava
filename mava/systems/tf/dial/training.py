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

# TODO (Kevin): implement DIAL trainer
# Helper resources
#   - single agent dqn learner in acme:
#           https://github.com/deepmind/acme/blob/master/acme/agents/tf/dqn/learning.py
#   - multi-agent ddpg trainer in mava: mava/systems/tf/maddpg/trainer.py

"""DIAL trainer implementation."""
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava


class DIALTrainer(mava.Trainer):
    """DIAL trainer.
    This is the trainer component of a DIAL system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        networks: Dict[str, snt.Module],
        target_network: Dict[str, snt.Module],
        discount: float,
        huber_loss_parameter: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        shared_weights: bool = True,
        importance_sampling_exponent: float = None,
        policy_optimizer: snt.Optimizer = None,
        replay_client: Optional[reverb.Client] = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "Checkpoints",
        max_gradient_norm: Optional[float] = None,
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          target_policy_network: the target policy (which lags behind the online
            policy).
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          dataset: dataset to learn from, whether fixed or from a replay buffer
            (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy
          target_observation_network: the target observation network.
          policy_optimizer: the optimizer to be applied to the (policy) loss.
          clipping: whether to clip gradients by global norm.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._agents = agents
        self._agent_types = agent_types
        self._shared_weights = shared_weights

        # Store online and target networks.
        self._policy_networks = networks
        self._target_policy_networks = target_network

        # self._observation_networks = observation_networks
        # self._target_observation_networks = target_observation_networks

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
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

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
                    "observation": self._observation_networks[agent_key],
                    "target_policy": self._target_policy_networks[agent_key],
                    "policy_optimizer": self._policy_optimizer,
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

        self._timestamp = None

    @tf.function
    def _update_target_networks(self) -> None:
        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (
                # *self._observation_networks[key].variables,
                *self._policy_networks[key].variables,
            )
            target_variables = (
                # *self._target_observation_networks[key].variables,
                *self._target_policy_networks[key].variables,
            )

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
            self._num_steps.assign_add(1)

    def _step(self) -> Dict[str, Dict[str, Any]]:
        # TODO Kevin: Implement DIAL trainer algorithm

        # Update the target networks
        self._update_target_networks()

        inputs = next(self._iterator)

        # [print(x, '\n') for x in inputs.data]
        o_tm1, a_tm1, e_tm1, r_t, d_t, s_t = inputs.data
        # o_tm1, a_tm1, e_tm1, r_t, d_t, o_t, e_t = inputs.data

        logged_losses: Dict[str, Dict[str, Any]] = {}
        return logged_losses

    def step(self) -> None:
        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        if self._timestamp:
            elapsed_time = timestamp - self._timestamp
        else:
            elapsed_time = 0
        self._timestamp = timestamp  # type: ignore

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint the networks.
        if len(self._system_checkpointer.keys()) > 0:
            for agent_key in self.unique_net_keys:
                checkpointer = self._system_checkpointer[agent_key]
                checkpointer.save()

        self._logger.write(fetches)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        variables: Dict[str, Dict[str, np.ndarray]] = {}
        for network_type in names:
            variables[network_type] = {}
            for agent in self.unique_net_keys:
                variables[network_type][agent] = tf2_utils.to_numpy(
                    self._system_network_variables[network_type][agent]
                )
        return variables
