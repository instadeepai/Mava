from typing import Dict, Tuple, Union

import dm_env
from absl.testing import parameterized

from mava.adders import reverb as reverb_adders
from tests.adders.adders_utils import MultiAgentAdderTestMixin
from tests.adders.sequence_adders_test_data import TEST_CASES

# TODO Clean this up, when using newer versions of acme.
try:
    from acme.adders.reverb.sequence import EndBehavior
except ImportError:
    from acme.adders.reverb.sequence import EndOfEpisodeBehavior as EndBehavior


class SequenceAdderTest(MultiAgentAdderTestMixin, parameterized.TestCase):
    @parameterized.named_parameters(*TEST_CASES)
    def test_adder(
        self,
        sequence_length: int,
        period: int,
        first: Union[Tuple, dm_env.TimeStep],
        steps: Tuple,
        expected_sequences: Tuple,
        agents: Dict,
        end_behavior: EndBehavior = EndBehavior.ZERO_PAD,
        repeat_episode_times: int = 1,
    ) -> None:
        adder = reverb_adders.ParallelSequenceAdder(
            self.client,
            sequence_length=sequence_length,
            period=period,
            end_of_episode_behavior=end_behavior,
        )
        super().run_test_adder(
            adder=adder,
            first=first,
            steps=steps,
            expected_items=expected_sequences,
            repeat_episode_times=repeat_episode_times,
            end_behavior=end_behavior,
            agents=agents,
        )
