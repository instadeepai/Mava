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

# # Adapted from
# https://github.com/deepmind/acme/blob/master/acme/adders/reverb/transition.py

"""TFRecord Transition adder.

This implements an single-step transition (SARS) adder which periodically
writes a buffer full of experience to disk for storage.
"""
from datetime import datetime
from typing import Dict, List, Union

import dm_env
from acme import types

from mava.adders.tfrecord.base import DEFAULT_SUBDIR, TFRecordParallelAdder


class TFRecordParallelTransitionAdder(TFRecordParallelAdder):
    """A TFRecord transition adder.

    Stores SARS tuples to TFRecord files on disk.
    """

    def __init__(
        self,
        transitions_per_file: int = 100_000,
        id: str = str(datetime.now()),
        subdir: str = DEFAULT_SUBDIR,
    ):
        """Initialise TFRecord Transition Adder.

        Args:
            transitions_per_file: number of transitions to store in each file.
            id: a string identifying this set of records.
            subdir: directory to which the records should be stored. Defualts to
                "~/mava/tfrecords/".

        """
        # A variable to store the last observation.
        self._observation: Union[None, types.NestedArray] = None

        # A buffer to hold transitions before
        # writing them to disk. Periodically cleared.
        self._buffer: List = []

    def _write(self) -> None:
        """Write all the experience in the buffer to a TFRecord."""
        pass

    def add_first(
        self, timestep: dm_env.TimeStep, extras: Dict[str, types.NestedArray] = {}
    ) -> None:
        """Record the first observation of a trajectory to the buffer.

        Args:
            timestep: dict of agents first observation.
            extras: dict of optional extras

        """
        raise NotImplementedError

    def add(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record an action and the following timestep.

        Args:
            actions: dict of agent actions.
            next_timestep: dict of agent observations.
            next_extras: dict of optional extras.

        """
        raise NotImplementedError
