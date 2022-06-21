import os
from typing import Any, Dict, Optional, Tuple

import dm_env
import numpy as np
import tensorflow as tf
import tree

from mava.adders.reverb.base import Trajectory
from mava.specs import MAEnvironmentSpec
from mava.utils.offline_utils import get_schema


class WriteSequence:
    def __init__(self, schema, sequence_length):
        self.schema = schema
        self.sequence_length = sequence_length
        self.numpy = tree.map_structure(
            lambda x: np.zeros(dtype=x.dtype, shape=(sequence_length, *x.shape)),
            schema,
        )
        self.t = 0

    def insert(self, agents, timestep, actions, next_timestep, extras):
        assert self.t < self.sequence_length
        for agent in agents:
            self.numpy[agent + "_observations"][self.t] = timestep.observation[
                agent
            ].observation

            self.numpy[agent + "_legal_actions"][self.t] = timestep.observation[
                agent
            ].legal_actions

            self.numpy[agent + "_actions"][self.t] = actions[agent]

            self.numpy[agent + "_rewards"][self.t] = next_timestep.reward[agent]

            self.numpy[agent + "_discounts"][self.t] = next_timestep.discount[agent]

        ## Extras
        # Zero padding mask
        self.numpy["zero_padding_mask"][self.t] = np.array(1, dtype=np.float32)

        # Global env state
        if "s_t" in extras:
            self.numpy["env_state"][self.t] = extras["s_t"]

        # increment t
        self.t += 1

    def zero_pad(self, agents, episode_return):
        # Maybe zero pad sequence
        while self.t < self.sequence_length:
            for agent in agents:
                for item in [
                    "_observations",
                    "_legal_actions",
                    "_actions",
                    "_rewards",
                    "_discounts",
                ]:
                    self.numpy[agent + item][self.t] = np.zeros_like(
                        self.numpy[agent + item][0]
                    )

                ## Extras
                # Zero-padding mask
                self.numpy["zero_padding_mask"][self.t] = np.zeros_like(
                    self.numpy["zero_padding_mask"][0]
                )

                # Global env state
                if "env_state" in self.numpy:
                    self.numpy["env_state"][self.t] = np.zeros_like(
                        self.numpy["env_state"][0]
                    )

            # Increment time
            self.t += 1

        self.numpy["episode_return"] = np.array(episode_return, dtype="float32")


class MAOfflineEnvironmentSequenceLogger:
    def __init__(
        self,
        environment,
        sequence_length: int,
        period: int,
        logdir: str = "./offline_env_logs",
        label: str = "",
        min_sequences_per_file: int = 1000,
    ):
        self._environment = environment
        self._schema = get_schema(self._environment)

        self._active_buffer = []
        self._write_buffer = []

        self._min_sequences_per_file = min_sequences_per_file
        self._sequence_length = sequence_length
        self._period = period

        self._logdir = logdir
        self._label = label
        os.makedirs(logdir, exist_ok=True)

        self._timestep: Optional[dm_env.TimeStep] = None
        self._extras: Optional[Dict] = None
        self._episode_return = None

        self._num_writes = 0
        self._timestep_ctr = 0

    def reset(self) -> Tuple[dm_env.TimeStep, Dict]:
        """Resets the env and log the first timestep.

        Returns:
            dm.env timestep, extras
        """
        timestep = self._environment.reset()

        if type(timestep) == tuple:
            self._timestep, self._extras = timestep
        else:
            self._timestep = timestep
            self._extras = {}

        self._episode_return = np.mean(list(self._timestep.reward.values()))
        self._active_buffer = []
        self._timestep_ctr = 0

        return self._timestep, self._extras

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[dm_env.TimeStep, Dict]:
        """Steps the env and logs timestep.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm.env timestep, extras
        """
        next_timestep = self._environment.step(actions)

        if type(next_timestep) == tuple:
            next_timestep, next_extras = next_timestep
        else:
            next_extras = {}

        self._episode_return += np.mean(list(next_timestep.reward.values()))

        # Log timestep
        self._log_timestep(
            self._timestep, self._extras, next_timestep, actions, self._episode_return
        )
        self._timestep = next_timestep
        self._extras = next_extras

        return self._timestep, self._extras

    def _log_timestep(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict,
        next_timestep: dm_env.TimeStep,
        actions: Dict,
        episode_return: float,
    ) -> None:
        if self._timestep_ctr % self._period == 0:
            self._active_buffer.append(
                WriteSequence(
                    schema=self._schema, sequence_length=self._sequence_length
                )
            )

        for write_sequence in self._active_buffer:
            if write_sequence.t < self._sequence_length:
                write_sequence.insert(
                    self.possible_agents, timestep, actions, next_timestep, extras
                )

        if next_timestep.last():
            for write_sequence in self._active_buffer:
                write_sequence.zero_pad(self.possible_agents, episode_return)
                self._write_buffer.append(write_sequence)
        if len(self._write_buffer) >= self._min_sequences_per_file:
            self._write()

        # Increment timestep counter
        self._timestep_ctr += 1

    def _write(self) -> None:
        filename = os.path.join(
            self._logdir, f"{self._label}_sequence_log_{self._num_writes}.tfrecord"
        )
        with tf.io.TFRecordWriter(filename, "GZIP") as file_writer:
            for write_sequence in self._write_buffer:

                # Convert numpy to tf.train features
                dict_of_features = tree.map_structure(
                    self._numpy_to_feature, write_sequence.numpy
                )

                # Create Example for writing
                features_for_example = tf.train.Features(feature=dict_of_features)
                example = tf.train.Example(features=features_for_example)

                # Write to file
                file_writer.write(example.SerializeToString())

        # Increment write counter
        self._num_writes += 1

        # Flush buffer and reset ctr
        self._write_buffer = []

    def _numpy_to_feature(self, np_array: np.ndarray):
        tensor = tf.convert_to_tensor(np_array)
        serialized_tensor = tf.io.serialize_tensor(tensor)
        bytes_list = tf.train.BytesList(value=[serialized_tensor.numpy()])
        feature_of_bytes = tf.train.Feature(bytes_list=bytes_list)
        return feature_of_bytes

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
