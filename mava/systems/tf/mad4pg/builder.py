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
import dataclasses
from typing import Any, Dict, Type

import sonnet as snt
from acme.utils import counting, loggers

from mava import core, specs
from mava.adders import reverb as reverb_adders
from mava.systems.tf import executors
from mava.systems.tf.mad4pg import training
from mava.systems.tf.maddpg.builder import MADDPGBuilder


@dataclasses.dataclass
class MAD4PGConfig:
    """Configuration options for the MAD4PG system.
    Args:
            environment_spec: description of the actions, observations, etc.
            policy_networks: the online (optimized) policies for each agent in
                the system.
            critic_networks: the online critic for each agent in the system.
            observation_networks: dictionary of optional networks to transform
                the observations before they are fed into any network.
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
            sigma: standard deviation of zero-mean, Gaussian exploration noise.
            clipping: whether to clip gradients by global norm.
            logger: logger object to be used by trainers.
            counter: counter object used to keep track of steps.
            checkpoint: boolean indicating whether to checkpoint the trainers.
            replay_table_name: string indicating what name to give the replay table."""

    environment_spec: specs.MAEnvironmentSpec
    policy_optimizer: snt.Optimizer
    critic_optimizer: snt.Optimizer
    shared_weights: bool = True
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    executor_variable_update_period: int = 1000
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: float = 32.0
    n_step: int = 5
    sequence_length: int = 20
    period: int = 20
    sigma: float = 0.3
    clipping: bool = True
    logger: loggers.Logger = None
    counter: counting.Counter = None
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


class MAD4PGBuilder(MADDPGBuilder):
    """Builder for MAD4PG which constructs individual components of the system."""

    """Defines an interface for defining the components of an RL system.
      Implementations of this interface contain a complete specification of a
      concrete RL system. An instance of this class can be used to build an
      RL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(
        self,
        config: MAD4PGConfig,
        trainer_fn: Type[
            training.BaseMAD4PGTrainer
        ] = training.DecentralisedMAD4PGTrainer,
        executor_fn: Type[core.Executor] = executors.FeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """Args:
        config: Configuration options for the MAD4PG system.
        trainer_fn: Trainer module to use."""

        super().__init__(
            config=config,
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
        )
