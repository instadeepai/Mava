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

"""Utilities for testing Reverb adders."""

from collections import Counter
from typing import Any, Dict, Sequence, Set

import dm_env
import numpy as np
import tensorflow as tf
import tree
from acme.adders.reverb import test_utils

# TODO Clean this up when using newer versions of acme.
try:
    from acme.adders.reverb.sequence import EndBehavior
except ImportError:
    from acme.adders.reverb.sequence import EndOfEpisodeBehavior as EndBehavior

from acme.specs import EnvironmentSpec
from acme.utils import tree_utils

from mava import specs
from mava.adders.reverb.base import ReverbParallelAdder
from mava.utils.wrapper_utils import parameterized_restart, parameterized_termination


def calc_nstep_return(
    r_t: Dict[str, float], discounts: list, rewards: list
) -> Dict[str, float]:
    """Function that calculates n_step_return as follows:
    R_{t:t+n} := r_t + d_t * r_{t+1} + ...    + d_t * ... * d_{t+n-2} * r_{t+n-1}.

    Args:
        r_t (Dict[str,float]): reward achieved from action a_t.
        discounts (list): list of discounts.
        rewards (list): list of rewards.

    Returns:
        A [Dict[str,float] with the return per agent.
    """
    reward = Counter(r_t)
    for index, r_tn in enumerate(rewards):
        d_t = discounts[index]
        return_t = {key: d_t[key] * r_tn.get(key, 0) for key in r_tn.keys()}
        reward.update(Counter(return_t))
    return dict(reward)


class MultiAgentAdderTestMixin(test_utils.AdderTestMixin):
    """A helper mixin for testing Reverb adders.
    Note that any test inheriting from this mixin must also inherit from something
    that provides the Python unittest assert methods.
    """

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


def make_trajectory(
    observations: np.ndarray, agents: Set[str] = {"agent_0", "agent_1", "agent_2"}
) -> Any:
    """Make a simple trajectory from a sequence of observations.
    Arguments:
        observations: a sequence of observations.
    Returns:
        a tuple (first, steps) where first contains the initial dm_env.TimeStep
        object and steps contains a list of (action, step) tuples. The length of
        steps is given by episode_length.
    """
    default_action = {agent: 0.0 for agent in agents}
    default_discount = {agent: 1.0 for agent in agents}
    final_discount = {agent: 0.0 for agent in agents}

    first = parameterized_restart(
        reward={agent: 0.0 for agent in agents},
        discount=default_discount,
        observation={agent: observations[0] for agent in agents},
    )

    middle = [
        (
            default_action,
            dm_env.transition(
                reward={agent: 0.0 for agent in agents},
                observation={agent: observation for agent in agents},
                discount=default_discount,
            ),
        )
        for observation in observations[1:-1]
    ]
    last = (
        default_action,
        parameterized_termination(
            reward={agent: 0.0 for agent in agents},
            observation={agent: observations[-1] for agent in agents},
            discount=final_discount,
        ),
    )
    return first, middle + [last]


def make_sequence(observations: np.ndarray) -> Any:
    """Create a sequence of timesteps of the form `first, [second, ..., last]`."""
    first, steps = make_trajectory(observations)

    agents = first.observation.keys()
    default_action = {agent: 0.0 for agent in agents}
    default_reward = {agent: 0.0 for agent in agents}
    final_discount = {agent: 0.0 for agent in agents}

    observation = first.observation
    sequence = []
    start_of_episode = True
    for action, timestep in steps:
        extras = ()
        sequence.append(
            (
                observation,
                action,
                timestep.reward,
                timestep.discount,
                start_of_episode,
                extras,
            )
        )
        observation = timestep.observation
        start_of_episode = False
    sequence.append(
        (observation, default_action, default_reward, final_discount, False, ())
    )
    return sequence
