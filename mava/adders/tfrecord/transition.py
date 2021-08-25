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
import os
from datetime import datetime
from typing import Dict, List

import dm_env
import numpy as np
import tensorflow as tf
from acme import types

from mava.adders.tfrecord.base import DEFAULT_SUBDIR, TFRecordParallelAdder
from mava.specs import MAEnvironmentSpec


class TFRecordParallelTransitionAdder(TFRecordParallelAdder):
    """A TFRecord transition adder.

    Stores SARS tuples to TFRecord files on disk.
    """

    def __init__(
        self,
        environment_spec: MAEnvironmentSpec,
        transitions_per_file: int = 100_000,
        id: str = str(datetime.now()),
        subdir: str = DEFAULT_SUBDIR,
    ):
        """Initialise TFRecord Transition Adder.

        Args:
            environment_spec: spec of the environment.
            transitions_per_file: number of transitions to store in each file.
            id: a string identifying this set of records.
            subdir: directory to which the records should be stored. Defualts to
                "~/mava/tfrecords/".

        """
        super().__init__(id, subdir)

        # Store env spec.
        self._environment_spec = environment_spec

        # A variable to store the last observation.
        self._observations: Dict = {}

        # A buffer to hold transitions before
        # writing them to disk. Periodically cleared.
        self._buffer: List = []
        self._max_buffer_size: int = transitions_per_file

        # Counters.
        self._write_ctr = 0
        self._buffer_ctr = 0

    def _bytes_feature(self, value: np.Array, dtype: str) -> tf.train.Feature:
        """Returns a bytes_list from a string / byte."""
        value = tf.convert_to_tensor(value, dtype=dtype)
        value = tf.io.serialize_tensor(value).numpy()
        bytes_list = tf.train.BytesList(value=[value])
        return tf.train.Feature(bytes_list=bytes_list)

    def _write(self) -> None:
        """Write all the experience in the buffer to a TFRecord."""
        if self._buffer_ctr >= self._max_buffer_size:
            filename = str(self._write_ctr) + ".tfrecord"
            path = os.path.join(self._subdir, filename)

            # Create writter.
            writer = tf.io.TFRecordWriter(path)

            # Write the content of the buffer.
            for transition in self._buffer:
                writer.write(transition.SerializeToString())

            # Close the writer.
            writer.close()

            # Increment write counter.
            self._write_ctr += 1

            # Clear buffer.
            self._buffer = []
            self._buffer_ctr = 0

    def add_first(
        self, timestep: dm_env.TimeStep, extras: Dict[str, types.NestedArray] = {}
    ) -> None:
        """Record the first observation of a trajectory to the buffer.

        Args:
            timestep: dict of agents first observation.
            extras: dict of optional extras

        """
        self._observations = timestep.observation

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
        # Get agent IDs and specs.
        agents = self._environment_spec.get_agent_ids()
        agent_specs = self._environment_spec.get_agent_specs()

        # Loop through agents and add tensors to
        # transition dict.
        transition: Dict = {}
        for agent in agents:

            # Store observation.
            key = "obs_" + agent
            observation = self._observations[agent].observation
            dtype = agent_specs[agent].observations.dtype
            transition[key] = self._bytes_feature(observation, dtype)

            # Store action.
            key = "act_" + agent
            action = actions[agent]
            dtype = agent_specs[agent].actions.dtype
            transition[key] = self._bytes_feature(action, dtype)

            # Store reward.
            key = "rew_" + agent
            reward = next_timestep.reward[agent]
            dtype = agent_specs[agent].rewards.dtype
            transition[key] = self._bytes_feature(reward, dtype)

            # Store next observation.
            key = "nob_" + agent
            next_observation = next_timestep.observation[agent].observation
            dtype = agent_specs[agent].observations.dtype
            transition[key] = self._bytes_feature(next_observation, dtype)

            # Store discount.
            key = "dis_" + agent
            discount = next_timestep.discount[agent]
            dtype = agent_specs[agent].discounts.dtype
            transition[key] = self._bytes_feature(discount, dtype)

        # Create TFExample
        transition = tf.train.Example(features=tf.train.Features(feature=transition))

        # Append the transition to the buffer.
        self._buffer.append(transition)
        self._buffer_ctr += 1

        # Update observation. Critical!!
        self._observation = next_timestep.observation

        # Maybe write to disk.
        self._write()
