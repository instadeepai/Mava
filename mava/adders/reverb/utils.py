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

import tree
from acme import types
from acme.adders.reverb import utils as acme_utils

from mava.adders.reverb import base


def final_step_like(
    step: base.Step, next_observations: types.NestedArray, next_extras: dict = None
) -> base.Step:
    """Return a list of steps with the final step zero-filled."""
    # Make zero-filled components so we can fill out the last step.
    zero_action, zero_reward, zero_discount = tree.map_structure(
        acme_utils.zeros_like, (step.actions, step.rewards, step.discounts)
    )

    return base.Step(
        observations=next_observations,
        actions=zero_action,
        rewards=zero_reward,
        discounts=zero_discount,
        start_of_episode=False,
        extras=next_extras
        if next_extras
        else tree.map_structure(acme_utils.zeros_like, step.extras),
    )
