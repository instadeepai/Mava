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

"""MAD4PG system implementation."""
from typing import Callable, Dict, Optional, Type, Union

import dm_env
import sonnet as snt
from acme import specs as acme_specs

from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedQValueActorCritic
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf.mad4pg import training
from mava.systems.tf.maddpg.execution import MADDPGFeedForwardExecutor
from mava.systems.tf.maddpg.system import MADDPG
from mava.utils.loggers import MavaLogger


class MAD4PG(MADDPG):
    """MAD4PG system.
    This implements a single-process D4PG system. This is an actor-critic based
    system that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policies of each agent
    (and as a result the behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[
            DecentralisedQValueActorCritic
        ] = DecentralisedQValueActorCritic,
        trainer_fn: Union[
            Type[training.MAD4PGBaseTrainer],
            Type[training.MAD4PGBaseRecurrentTrainer],
        ] = training.MAD4PGDecentralisedTrainer,
        executor_fn: Type[core.Executor] = MADDPGFeedForwardExecutor,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_averaging: bool = False,
        target_update_period: int = 100,
        target_update_rate: Optional[float] = None,
        executor_variable_update_period: int = 1000,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: float = 32.0,
        policy_optimizer: Union[
            snt.Optimizer, Dict[str, snt.Optimizer]
        ] = snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        n_step: int = 5,
        sequence_length: int = 20,
        period: int = 20,
        sigma: float = 0.3,
        max_gradient_norm: float = None,
        max_executor_steps: int = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
    ):
        """Initialize the system.
        Args:
            environment_factory: Callable to instantiate an environment
                on a compute node.
            network_factory: Callable to instantiate system networks on a compute node.
            logger_factory: Callable to instantiate a system logger on a compute node.
            architecture: system architecture, e.g. decentralised or centralised.
            trainer_fn: training type associated with executor and architecture,
                e.g. centralised training.
            executor_fn: executor type for example feedforward or recurrent.
            num_executors: number of executor processes to run in parallel.
            num_caches: number of trainer node caches.
            environment_spec: description of the actions, observations, etc.
            shared_weights: set whether agents should share network weights.
            discount: discount to use for TD updates.
            batch_size: batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_update_period: number of learner steps to perform before updating
              the target networks.
            min_replay_size: minimum replay size before updating.
            max_replay_size: maximum replay size.
            samples_per_insert: number of samples to take from replay for every insert
              that is made.
            n_step: number of steps to squash into a single transition.
            sequence_length: Length of the sequences to use in recurrent
            training (if using recurrence).
            period: Overlapping period of sequences used in recurrent
            training (if using recurrence).
            sigma: standard deviation of zero-mean, Gaussian exploration noise.
            max_gradient_norm:
            logger: logger object to be used by trainers.
            counter: counter object used to keep track of steps.
            checkpoint: boolean indicating whether to checkpoint the trainers.
            checkpoint_subpath: directory for checkpoints.
            replay_table_name: string indicating what name to give the replay table.
            train_loop_fn: loop for training.
            eval_loop_fn: loop for evaluation.
            policy_optimizer: the optimizer to be applied to the policy loss.
                This can be a single optimizer or an optimizer per agent key.
            critic_optimizer: the optimizer to be applied to the critic loss.
        """
        super().__init__(
            environment_factory=environment_factory,
            network_factory=network_factory,
            logger_factory=logger_factory,
            architecture=architecture,
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            num_executors=num_executors,
            num_caches=num_caches,
            environment_spec=environment_spec,
            shared_weights=shared_weights,
            discount=discount,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            target_update_period=target_update_period,
            executor_variable_update_period=executor_variable_update_period,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            samples_per_insert=samples_per_insert,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            n_step=n_step,
            sequence_length=sequence_length,
            period=period,
            sigma=sigma,
            max_gradient_norm=max_gradient_norm,
            max_executor_steps=max_executor_steps,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            logger_config=logger_config,
            train_loop_fn=train_loop_fn,
            eval_loop_fn=eval_loop_fn,
            train_loop_fn_kwargs=train_loop_fn_kwargs,
            eval_loop_fn_kwargs=eval_loop_fn_kwargs,
            target_averaging=target_averaging,
            target_update_rate=target_update_rate,
        )
