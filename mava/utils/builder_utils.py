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
from typing import Callable, Dict, Optional, Union

from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationScheduler,
    BaseExplorationTimestepScheduler,
    ConstantScheduler,
)


def initialize_epsilon_schedulers(
    exploration_schedules: Dict[
        str,
        Union[
            BaseExplorationScheduler,
            BaseExplorationTimestepScheduler,
            ConstantScheduler,
        ],
    ],
    action_selectors: Dict[str, Callable],
    agent_net_keys: Dict[str, str],
    seed: Optional[int] = None,
) -> Dict:
    """Function that initializes action selectors.

    Args:
        exploration_schedules : epsilon decay schedule per agent.
        action_selectors : dict containing the action selector functions
            (e.g. LinearExplorationTimestepScheduler) per network.
        agent_net_keys: specifies what network each agent uses.
        evaluator: boolean indicator if the executor is used for
            for evaluation only.
        seed: seed for reproducible sampling.

    Returns:
        dict with initialized action selectors with schedules.
    """
    # Pass scheduler and initialize action selectors
    action_selectors_with_scheduler: Dict = dict.fromkeys(exploration_schedules, [])
    for agent in exploration_schedules.keys():
        schedule = exploration_schedules[agent]
        network_for_agent = agent_net_keys[agent]
        action_selectors_with_scheduler[agent] = action_selectors[network_for_agent](
            schedule, seed=seed
        )

    return action_selectors_with_scheduler
