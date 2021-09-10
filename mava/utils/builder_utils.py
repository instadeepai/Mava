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
from typing import Callable, Dict, Optional, Type, Union

from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationScheduler,
    BaseExplorationTimestepScheduler,
    ConstantScheduler,
)


def initialize_epsilon_schedulers(
    exploration_scheduler_fn: Union[
        Type[BaseExplorationScheduler], Type[BaseExplorationTimestepScheduler]
    ],
    action_selectors: Dict[str, Callable],
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: Optional[float] = None,
    epsilon_decay_steps: Optional[int] = None,
) -> Dict:
    """Function that initializes action selectors.

    Args:
        exploration_scheduler_fn : epsilon decay exploration function.
        action_selectors : dict containing the action selector functions
            (e.g. LinearExplorationTimestepScheduler) per network.
        epsilon_start : initial eps value.
        epsilon_min : final eps value.
        epsilon_decay : eps decay, if using a BaseExplorationScheduler.
        epsilon_decay_steps : eps decay steps,
            if using a BaseExplorationTimestepScheduler.

    Raises:
        ValueError: incorrect exploration_scheduler_fn.

    Returns:
        dict with initialized action selectors.
    """
    # Pass scheduler and initialize action selectors
    action_selectors_return = {}
    for network, action_selector_fn in action_selectors.items():
        scheduler: Union[
            BaseExplorationScheduler,
            BaseExplorationTimestepScheduler,
            ConstantScheduler,
        ]
        # If eps start and min are the same, it is more efficient to use
        # ConstantScheduler
        if epsilon_start == epsilon_min:
            scheduler = ConstantScheduler(epsilon_start=epsilon_start)

        elif issubclass(exploration_scheduler_fn, BaseExplorationScheduler):
            assert epsilon_decay is not None
            scheduler = exploration_scheduler_fn(
                epsilon_start=epsilon_start,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
            )

        elif issubclass(exploration_scheduler_fn, BaseExplorationTimestepScheduler):
            assert epsilon_decay_steps is not None
            scheduler = exploration_scheduler_fn(
                epsilon_start=epsilon_start,
                epsilon_min=epsilon_min,
                epsilon_decay_steps=epsilon_decay_steps,
            )

        else:
            raise ValueError(
                f"Invalid exploration_scheduler_fn: \
                     {exploration_scheduler_fn}"
            )

        action_selectors_return[network] = action_selector_fn(scheduler)
    return action_selectors_return
