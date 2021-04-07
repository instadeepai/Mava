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


"""MADDPG trainer implementation."""
import copy
from typing import Dict, List, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.utils import counting, loggers

from mava.systems.tf.maddpg import training_base


class MADDPGTrainer(training_base.MADDPGTrainer):
    """MADDPG trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        target_policy_networks: Dict[str, snt.Module],
        target_critic_networks: Dict[str, snt.Module],
        discount: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_network: the target critic.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          dataset: dataset to learn from, whether fixed or from a replay buffer
            (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy and the critic.
          target_observation_network: the target observation network.
          policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
          critic_optimizer: the optimizer to be applied to the critic loss.
          clipping: whether to clip gradients by global norm.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """
        super().__init__(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )

    @tf.function
    def _get_critic_feed(
        self,
        o_tm1_trans: Dict[str, np.ndarray],
        o_t_trans: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        a_t: Dict[str, np.ndarray],
        e_t: Dict[str, np.array],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        # State based
        o_tm1_feed = e_t["s_tm1"]
        o_t_feed = e_t["s_t"]
        a_tm1_feed = tf.stack([x for x in a_tm1.values()], 1)
        a_t_feed = tf.stack([x for x in a_t.values()], 1)

        return o_tm1_feed, o_t_feed, a_tm1_feed, a_t_feed

    @tf.function
    def _get_dpg_feed(
        self,
        a_t: Dict[str, np.ndarray],
        dpg_a_t: np.ndarray,
        agent: str,
    ) -> tf.Tensor:
        # StateBased DPG
        # Note (dries): Copy has to be made because the input
        # variables cannot be changed.
        dpg_a_t_feed = copy.copy(a_t)
        dpg_a_t_feed[agent] = dpg_a_t

        return dpg_a_t_feed
