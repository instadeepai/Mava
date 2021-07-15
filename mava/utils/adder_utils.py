from typing import Dict, Sequence, Union

from acme.adders.reverb import base
from acme.tf import utils as tf2_utils


def calculate_priorities(
    priority_fns: base.PriorityFnMapping, steps: Union[base.Step, Sequence[base.Step]]
) -> Dict[str, float]:
    """Helper used to calculate the priority of a sequence of steps."""

    if isinstance(steps, list):
        steps = tf2_utils.stack_sequence_fields(steps)

    return {
        table: (priority_fn(steps) if priority_fn else 1.0)
        for table, priority_fn in priority_fns.items()
    }
