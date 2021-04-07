import tree
from acme import types
from acme.adders.reverb import utils as acme_utils

from mava.adders.reverb import base


def final_step_like(step: base.Step, next_observation: types.NestedArray) -> base.Step:
    """Return a list of steps with the final step zero-filled."""
    # Make zero-filled components so we can fill out the last step.
    zero_action, zero_reward, zero_discount, zero_extras = tree.map_structure(
        acme_utils.zeros_like, (step.actions, step.rewards, step.discounts, step.extras)
    )

    # Return a final step that only has next_observation.
    return base.Step(
        observations=next_observation,
        actions=zero_action,
        rewards=zero_reward,
        discounts=zero_discount,
        start_of_episode=False,
        extras=zero_extras,
    )
