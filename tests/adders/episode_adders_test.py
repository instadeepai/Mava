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
        """Test episode adders

        Args:
            max_sequence_length: The maximum seuqence length of the adder
            first: The first `dm_env.TimeStep` that is used to call
                `base.ReverbAdder.add_first()`.
            steps: A sequence of (action, timestep) tuples that are passed to
                `base.ReverbAdder.add()`.
            expected_sequences: The sequence of items that are expected to be created
                by calling the adder's `add_first()` method on `first` and `add()`
                on all of the elements in `steps`.
            agents: Dict containing agent names, e.g.
                agents = {"agent_0", "agent_1", "agent_2"}.
            repeat_episode_times: How many times to run an episode.
                end_behavior: How end of episode should be handled.

        Returns:
            None
        """
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
