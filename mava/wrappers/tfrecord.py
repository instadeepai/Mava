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

"""Wraps an executor so that experience gets stored to TFRecord."""
from typing import Any, Dict

import dm_env
from acme import types

from mava.adders.tfrecord import TFRecordParallelAdder
from mava.systems.tf.executors import FeedForwardExecutor


class TFRecordWrapper:
    def __init__(
        self,
        executor: FeedForwardExecutor,
        tfrecord_adder: TFRecordParallelAdder,
    ):
        """A class to wrap an executor to store experience to disk.

        Args:
            executor: A mava executor.
            tfrecord_adder: A TFRecord adder to write experience to disk.
        """
        self._executor = executor
        self._adder = tfrecord_adder

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Wrapped observe first function.

        Args:
            timestep: first time step.
            extras: extra info to store. Defaults to {}.
        """
        # Add transition to TFRecord.
        self._adder.add_first(timestep, extras)

        # Call observe function of the executor.
        self._executor.observe_first(timestep, extras)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Wraped observe function.

        Args:
            actions: agent actions.
            next_timestep: next timestep.
            next_extras: extra information from the next timestep.
                Defaults to {}.
        """
        # Add transition to TFRecord.
        self._adder.add(actions, next_timestep, next_extras)

        # Call observe function of the executor.
        self._executor.observe(actions, next_timestep, next_extras)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying executor."""
        return getattr(self._executor, name)
