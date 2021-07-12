"""Utilities for testing Reverb adders."""

from typing import Any, Dict, Sequence

import dm_env
import numpy as np
import tensorflow as tf
import tree
from acme.adders.reverb import test_utils
from acme.adders.reverb.sequence import EndOfEpisodeBehavior as EndBehavior
from acme.specs import EnvironmentSpec
from acme.utils import tree_utils

from mava import specs
from mava.adders.reverb.base import ReverbParallelAdder


class MAAdderTestMixin(test_utils.AdderTestMixin):
    #   """A helper mixin for testing Reverb adders.
    #   Note that any test inheriting from this mixin must also inherit from something
    #   that provides the Python unittest assert methods.
    #   """
    def run_test_adder(
        self,
        adder: ReverbParallelAdder,
        first: dm_env.TimeStep,
        steps: Sequence,
        expected_items: Sequence[Any],
        agents: Dict,
        stack_sequence_fields: bool = True,
        repeat_episode_times: int = 1,
        end_behavior: EndBehavior = EndBehavior.ZERO_PAD,
    ) -> None:

        """
        Runs a unit test case for the adder.
        Args:
                adder: The instance of `base.ReverbAdder` that is being tested.
                first: The first `dm_env.TimeStep` that is used to call
                    `base.ReverbAdder.add_first()`.
                steps: A sequence of (action, timestep) tuples that are passed to
                    `base.ReverbAdder.add()`.
                expected_items: The sequence of items that are expected to be created
                    by calling the adder's `add_first()` method on `first` and `add()`
                    on all of the elements in `steps`.
                agents: Dict containing agent names, e.g.
                    agents = {"agent_0", "agent_1", "agent_2"}.
                repeat_episode_times: How many times to run an episode.
                    end_behavior: How end of episode should be handled.
        """

        if not steps:
            raise ValueError("At least one step must be given.")

        agent_specs = {}
        for agent in agents:
            agent_specs[agent] = EnvironmentSpec(
                observations=test_utils._numeric_to_spec(
                    steps[0][1].observation[agent]
                ),
                actions=test_utils._numeric_to_spec(steps[0][0][agent]),
                rewards=test_utils._numeric_to_spec(steps[0][1].reward[agent]),
                discounts=test_utils._numeric_to_spec(steps[0][1].discount[agent]),
            )

        has_extras = len(steps[0]) >= 3

        if has_extras:
            extras_spec = tree.map_structure(test_utils._numeric_to_spec, steps[0][2])
        else:
            extras_spec = {}

        ma_spec = specs.MAEnvironmentSpec(
            environment=None, specs=agent_specs, extra_specs=extras_spec
        )

        signature = adder.signature(ma_spec, extras_spec=extras_spec)

        for episode_id in range(repeat_episode_times):
            # Add all the data up to the final step.
            # Check if first step has extras
            if type(first) == tuple:
                first, extras = first
                adder.add_first(first, extras)
            else:
                adder.add_first(first)

            # timestep: dm_env.TimeStep, extras: Dict[str, types.NestedArray]
            for step in steps[:-1]:
                action, ts = step[0], step[1]

                if has_extras:
                    extras = step[2]
                else:
                    extras = ()

                adder.add(action, next_timestep=ts, next_extras=extras)

            # Add the final step.
            adder.add(*steps[-1])

        # Ending the episode should close the writer. No new writer should yet have
        # been created as it is constructed lazily.
        if end_behavior is not EndBehavior.CONTINUE:
            self.assertEqual(self.client.writer.num_episodes, repeat_episode_times)

        # Make sure our expected and observed data match.
        observed_items = [p[2] for p in self.client.writer.priorities]

        # Check matching number of items.
        self.assertEqual(len(expected_items), len(observed_items))

        # Check items are matching according to numpy's almost_equal.
        for expected_item, observed_item in zip(expected_items, observed_items):
            if stack_sequence_fields:
                expected_item = tree_utils.stack_sequence_fields(expected_item)

            # Set check_types=False because we check them below.
            tree.map_structure(
                np.testing.assert_array_almost_equal,
                expected_item,
                tuple(observed_item),
                check_types=False,
            )

        # Make sure the signature matches was is being written by Reverb.
        def _check_signature(spec: tf.TensorSpec, value: np.ndarray) -> None:
            self.assertTrue(spec.is_compatible_with(tf.convert_to_tensor(value)))

        # Check the last transition's signature.
        tree.map_structure(_check_signature, signature, observed_items[-1])
