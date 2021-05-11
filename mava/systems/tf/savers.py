"""Utility classes for saving model checkpoints and snapshots
 that leverages acme's checkpointer."""

from typing import Any, Union

from acme import core
from acme.tf import savers


# Classes adapted from https://github.com/deepmind/acme/blob/master/acme/tf/savers.py
class Checkpointer(savers.Checkpointer):
    def __init__(self, directory: str = "~/mava/", **kwargs: Any) -> None:
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
        super().__init__(directory=directory, **kwargs)
