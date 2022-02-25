from typing import Dict, Tuple, Union

import dm_env
from absl.testing import parameterized

from mava.adders import reverb as reverb_adders
from tests.adders.adders_utils import MultiAgentAdderTestMixin
from tests.adders.episode_adders_test_data import TEST_CASES


class EpisodeAdderTest(MultiAgentAdderTestMixin, parameterized.TestCase):
    @parameterized.named_parameters(*TEST_CASES)
    def test_adder(
        self,
        max_sequence_length: int,
        first: Union[Tuple, dm_env.TimeStep],
        steps: Tuple,
        expected_sequences: Tuple,
        agents: Dict,
        repeat_episode_times: int = 1,
    ) -> None:
        adder = reverb_adders.ParallelEpisodeAdder(
            self.client,
            max_sequence_length=max_sequence_length,
        )
        super().run_test_adder(
            adder=adder,
            first=first,
            steps=steps,
            expected_items=expected_sequences,
            repeat_episode_times=repeat_episode_times,
            agents=agents,
        )
