# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Transition adder unit tests"""

from typing import Dict, Tuple, Union

import dm_env
from absl.testing import parameterized

from mava.adders import reverb as reverb_adders
from tests.adders.adders_utils import MultiAgentAdderTestMixin
from tests.adders.transition_adders_test_data import TEST_CASES


class TestParallelNStepTransitionAdder(
    MultiAgentAdderTestMixin, parameterized.TestCase
):
    """Parallel nstep transition adder testing class"""

    @parameterized.named_parameters(*TEST_CASES)
    def test_transition_adder(
        self,
        n_step: int,
        discount: float,
        first: Union[Tuple, dm_env.TimeStep],
        steps: Tuple,
        expected_transitions: Tuple,
        agents: Dict,
    ) -> None:
        """Test sequence adders

        Args:
            n_step: The "N" in N-step transition. See the class docstring for the
            precise definition of what an N-step transition is. `n_step` must be at
            least 1, in which case we use the standard one-step transition, i.e.
            (s_t, a_t, r_t, d_t, s_t+1, e_t).
            discount: Discount factor to apply.
            first: The first `dm_env.TimeStep` that is used to call
                `base.ReverbAdder.add_first()`.
            steps: A sequence of (action, timestep) tuples that are passed to
                `base.ReverbAdder.add()`.
            expected_sequences: The sequence of items that are expected to be created
                by calling the adder's `add_first()` method on `first` and `add()`
                on all of the elements in `steps`.
            agents: Dict containing agent names, e.g.
                agents = {"agent_0", "agent_1", "agent_2"}.
            end_behavior: How end of episode should be handled.
            repeat_episode_times: How many times to run an episode.
                end_behavior: How end of episode should be handled.

        Returns:
            None
        """
        adder = reverb_adders.ParallelNStepTransitionAdder(
            self.client, n_step, discount
        )
        super().run_test_adder(
            adder=adder,
            first=first,
            steps=steps,
            expected_items=expected_transitions,
            agents=agents,
            stack_sequence_fields=False,
        )
