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

"""MADDPG configuration."""

import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Type, Union

# TODO (Arnu): align config with MADDPG system config function


@dataclasses.dataclass
class MADDPGConfig:
    """Configuration options for the MADDPG system.

    Args:
        environment_spec: description of the action and observation spaces etc. for
            each agent in the system.
        policy_optimizer: optimizer(s) for updating policy networks.
        critic_optimizer: optimizer for updating critic networks.
        num_executors: number of parallel executors to use.
        agent_net_keys: specifies what network each agent uses.
        trainer_networks: networks each trainer trains on.
        table_network_config: Networks each table (trainer) expects.
        network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
        net_keys_to_ids: mapping from net_key to network id.
        unique_net_keys: list of unique net_keys.
        checkpoint_minute_interval: The number of minutes to wait between
            checkpoints.
        discount: discount to use for TD updates.
        batch_size: batch size for updates.
        prefetch_size: size to prefetch from replay.
        target_averaging: whether to use polyak averaging for target network updates.
        target_update_period: number of steps before target networks are updated.
        target_update_rate: update rate when using averaging.
        executor_variable_update_period: the rate at which executors sync their
            paramters with the trainer.
        min_replay_size: minimum replay size before updating.
        max_replay_size: maximum replay size.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        n_step: number of steps to include prior to boostrapping.
        sequence_length: recurrent sequence rollout length.
        period: consecutive starting points for overlapping rollouts across a sequence.
        max_gradient_norm: value to specify the maximum clipping value for the gradient
            norm during optimization.
        logger: logger to use.
        checkpoint: boolean to indicate whether to checkpoint models.
        checkpoint_subpath: subdirectory specifying where to store checkpoints.
        termination_condition: An optional terminal condition can be provided
            that stops the program once the condition is satisfied. Available options
            include specifying maximum values for trainer_steps, trainer_walltime,
            evaluator_steps, evaluator_episodes, executor_episodes or executor_steps.
            E.g. termination_condition = {'trainer_steps': 100000}.
        learning_rate_scheduler_fn: dict with two functions/classes (one for the
                policy and one for the critic optimizer), that takes in a trainer
                step t and returns the current learning rate,
                e.g. {"policy": policy_lr_schedule ,"critic": critic_lr_schedule}.
                See
                examples/debugging/simple_spread/feedforward/decentralised/run_maddpg_lr_schedule.py
                for an example.
        evaluator_interval: An optional condition that is used to
            evaluate/test system performance after [evaluator_interval]
            condition has been met.
    """

    environment_spec: specs.MAEnvironmentSpec
    policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]]
    critic_optimizer: snt.Optimizer
    num_executors: int
    agent_net_keys: Dict[str, str]
    trainer_networks: Dict[str, List]
    table_network_config: Dict[str, List]
    network_sampling_setup: List
    net_keys_to_ids: Dict[str, int]
    unique_net_keys: List[str]
    checkpoint_minute_interval: int
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_averaging: bool = False
    target_update_period: int = 100
    target_update_rate: Optional[float] = None
    executor_variable_update_period: int = 1000
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: Optional[float] = 32.0
    n_step: int = 5
    sequence_length: int = 20
    period: int = 20
    bootstrap_n: int = 10
    max_gradient_norm: Optional[float] = None
    logger: loggers.Logger = None
    counter: counting.Counter = None
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    termination_condition: Optional[Dict[str, int]] = None
    evaluator_interval: Optional[dict] = None
    learning_rate_scheduler_fn: Optional[Any] = None
