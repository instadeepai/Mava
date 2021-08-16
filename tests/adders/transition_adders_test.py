from typing import Dict, Tuple, Union

import dm_env
from absl.testing import parameterized

from mava.adders import reverb as reverb_adders
from tests.adders.adders_utils import MultiAgentAdderTestMixin
from tests.adders.transition_adders_test_data import TEST_CASES


class TestParallelNStepTransitionAdder(
    MultiAgentAdderTestMixin, parameterized.TestCase
):
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
