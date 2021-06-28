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
        """[summary]

        Args:
            environment_factory (Callable[[bool], dm_env.Environment]): [description]
            network_factory (Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]]): [description]
            logger_factory (Callable[[str], MavaLogger], optional): [description]. Defaults to None.
            architecture (Type[ DecentralisedQValueActorCritic ], optional): [description]. Defaults to DecentralisedQValueActorCritic.
            trainer_fn (Union[ Type[training.MAD4PGBaseTrainer], Type[training.MAD4PGBaseRecurrentTrainer], ], optional): [description]. Defaults to training.MAD4PGDecentralisedTrainer.
            executor_fn (Type[core.Executor], optional): [description]. Defaults to MADDPGFeedForwardExecutor.
            num_executors (int, optional): [description]. Defaults to 1.
            num_caches (int, optional): [description]. Defaults to 0.
            environment_spec (mava_specs.MAEnvironmentSpec, optional): [description]. Defaults to None.
            shared_weights (bool, optional): [description]. Defaults to True.
            discount (float, optional): [description]. Defaults to 0.99.
            batch_size (int, optional): [description]. Defaults to 256.
            prefetch_size (int, optional): [description]. Defaults to 4.
            target_averaging (bool, optional): [description]. Defaults to False.
            target_update_period (int, optional): [description]. Defaults to 100.
            target_update_rate (Optional[float], optional): [description]. Defaults to None.
            executor_variable_update_period (int, optional): [description]. Defaults to 1000.
            min_replay_size (int, optional): [description]. Defaults to 1000.
            max_replay_size (int, optional): [description]. Defaults to 1000000.
            samples_per_insert (float, optional): [description]. Defaults to 32.0.
            policy_optimizer (Union[ snt.Optimizer, Dict[str, snt.Optimizer] ], optional): [description]. Defaults to snt.optimizers.Adam(learning_rate=1e-4).
            critic_optimizer (snt.Optimizer, optional): [description]. Defaults to snt.optimizers.Adam(learning_rate=1e-4).
            n_step (int, optional): [description]. Defaults to 5.
            sequence_length (int, optional): [description]. Defaults to 20.
            period (int, optional): [description]. Defaults to 20.
            sigma (float, optional): [description]. Defaults to 0.3.
            max_gradient_norm (float, optional): [description]. Defaults to None.
            max_executor_steps (int, optional): [description]. Defaults to None.
            checkpoint (bool, optional): [description]. Defaults to True.
            checkpoint_subpath (str, optional): [description]. Defaults to "~/mava/".
            logger_config (Dict, optional): [description]. Defaults to {}.
            train_loop_fn (Callable, optional): [description]. Defaults to ParallelEnvironmentLoop.
            eval_loop_fn (Callable, optional): [description]. Defaults to ParallelEnvironmentLoop.
            train_loop_fn_kwargs (Dict, optional): [description]. Defaults to {}.
            eval_loop_fn_kwargs (Dict, optional): [description]. Defaults to {}.
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
