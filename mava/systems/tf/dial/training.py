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
from typing import Dict, List, Optional

import reverb
import sonnet as snt
import tensorflow as tf
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
        shared_weights: bool = False,
        importance_sampling_exponent: float = None,
        policy_optimizer: snt.Optimizer = None,
        replay_client: Optional[reverb.Client] = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
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
        self._shared_weights = shared_weights

        # Store online and target networks.
        self._networks = networks
        self._target_networks = target_network
