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

"""Utility classes for saving model checkpoints and snapshots
 that leverages acme's checkpointer."""

from typing import Any, Union

from acme import core
from acme.tf import savers


# Classes adapted from https://github.com/deepmind/acme/blob/master/acme/tf/savers.py
class Checkpointer(savers.Checkpointer):
    def __init__(self, directory: str = "~/mava/", **kwargs: Any) -> None:
        """Initialise checkpointer.

        Args:
            directory (str, optional): directory to store checkpoints. Defaults
                to "~/mava/".
        """

        super().__init__(directory=directory, add_uid=False, **kwargs)


class CheckpointingRunner(savers.CheckpointingRunner):
    """Wrap an object and expose a run method which checkpoints periodically.

    This internally creates a Checkpointer around `wrapped` object and exposes
    all of the methods of `wrapped`. Additionally, anay `**kwargs` passed to the
    runner are forwarded to the internal Checkpointer.
    """

    def __init__(
        self,
        wrapped: Union[
            savers.Checkpointable, core.Saveable, core.Learner, savers.TFSaveable
        ],
        *,
        time_delta_minutes: int = 30,
        **kwargs: Any,
    ):
        """Initialise checkpoint runner.

        Args:
            wrapped (Union[ savers.Checkpointable, core.Saveable, core.Learner,
                savers.TFSaveable ]): wrapped object for checkpointing.
            time_delta_minutes (int, optional): time between consecutive checkpoints.
                Defaults to 30.
        """

        if isinstance(wrapped, savers.TFSaveable):
            # If the object to be wrapped exposes its TF State, checkpoint that.
            objects_to_save = wrapped.state
        else:
            # Otherwise checkpoint the wrapped object itself.
            objects_to_save = wrapped

        self._wrapped = wrapped
        self._time_delta_minutes = time_delta_minutes
        self._checkpointer = Checkpointer(
            objects_to_save={"wrapped": objects_to_save},
            time_delta_minutes=time_delta_minutes,
            **kwargs,
        )


class Snapshotter(savers.Snapshotter):
    """Convenience class for periodically snapshotting."""

    def __init__(self, directory: str = "~/mava/", **kwargs: Any) -> None:
        """Initialise snapshotter.

        Args:
            directory (str, optional): Directory in which to store snapshots. Defaults
                to "~/mava/".
        """

        super().__init__(directory=directory, **kwargs)
