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

"""MAD4PG system builder implementation."""

from typing import Any, Dict, Type, Union

from mava import core
from mava.systems.tf import executors
from mava.systems.tf.mad4pg import training
from mava.systems.tf.maddpg.builder import MADDPGBuilder, MADDPGConfig


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
        config: MADDPGConfig,
        trainer_fn: Union[
            Type[training.MAD4PGBaseTrainer],
            Type[training.MAD4PGBaseRecurrentTrainer],
        ] = training.MAD4PGDecentralisedTrainer,
        executor_fn: Type[core.Executor] = executors.FeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """[summary]

        Args:
            config (MADDPGConfig): [description]
            trainer_fn (Union[ Type[training.MAD4PGBaseTrainer], Type[training.MAD4PGBaseRecurrentTrainer], ], optional): [description]. Defaults to training.MAD4PGDecentralisedTrainer.
            executor_fn (Type[core.Executor], optional): [description]. Defaults to executors.FeedForwardExecutor.
            extra_specs (Dict[str, Any], optional): [description]. Defaults to {}.
        """

        super().__init__(
            config=config,
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
        )
