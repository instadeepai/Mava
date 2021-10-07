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

# # Adapted from
# https://github.com/deepmind/acme/blob/master/acme/adders/reverb/transition.py

"""Transition adders.

This implements an N-step transition adder which collapses trajectory sequences
into a single transition, simplifying to a simple transition adder when N=1.
"""
import copy
from typing import Dict, List, Optional

import numpy as np
import reverb
import tensorflow as tf
import tree
from acme.adders.reverb import utils as acme_utils
from acme.adders.reverb.transition import NStepTransitionAdder, _broadcast_specs
from acme.utils import tree_utils

from mava import specs as mava_specs
from mava import types
from mava import types as mava_types
from mava.adders.reverb import base
from mava.adders.reverb.base import ReverbParallelAdder


class ParallelNStepTransitionAdder(NStepTransitionAdder, ReverbParallelAdder):
    """An N-step transition adder.

    This will buffer a sequence of N timesteps in order to form a single N-step
    transition which is added to reverb for future retrieval.
    For N=1 the data added to replay will be a standard one-step transition which
    takes the form:
          (s_t, a_t, r_t, d_t, s_{t+1}, e_t)
    where:
      s_t = state observation at time t
      a_t = the action taken from s_t
      r_t = reward ensuing from action a_t
      d_t = environment discount ensuing from action a_t. This discount is
          applied to future rewards after r_t.
      e_t [Optional] = extra data that the agent persists in replay.
    For N greater than 1, transitions are of the form:
          (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+N}, e_t),
    where:
      s_t = State (observation) at time t.
      a_t = Action taken from state s_t.
      g = the additional discount, used by the agent to discount future returns.
      R_{t:t+n} = N-step discounted return, i.e. accumulated over N rewards:
            R_{t:t+n} := r_t + g * d_t * r_{t+1} + ...
                             + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}.
      D_{t:t+n}: N-step product of agent discounts g_i and environment
        "discounts" d_i.
            D_{t:t+n} := g^{n-1} * d_{t} * ... * d_{t+n-1},
        For most environments d_i is 1 for all steps except the last,
        i.e. it is the episode termination signal.
      s_{t+n}: The "arrival" state, i.e. the state at time t+n.
      e_t [Optional]: A nested structure of any 'extras' the user wishes to add.

    Notes:
      - At the beginning and end of episodes, shorter transitions are added.
        That is, at the beginning of the episode, it will add:
              (s_0 -> s_1), (s_0 -> s_2), ..., (s_0 -> s_n), (s_1 -> s_{n+1})
        And at the end of the episode, it will add:
              (s_{T-n+1} -> s_T), (s_{T-n+2} -> s_T), ... (s_{T-1} -> s_T).
      - We add the *first* `extra` of each transition, not the *last*, i.e.
          if extras are provided, we get e_t, not e_{t+n}.
    """

    def __init__(
        self,
        client: reverb.Client,
        n_step: int,
        discount: float,
        net_ids_to_keys: List[str] = None,
        table_network_config: Dict[str, List] = None,
        *,
        priority_fns: Optional[base.PriorityFnMapping] = None,
        max_in_flight_items: int = 5,
    ) -> None:
        """Creates an N-step transition adder.

        Args:
          client: A `reverb.Client` to send the data to replay through.
          n_step: The "N" in N-step transition. See the class docstring for the
            precise definition of what an N-step transition is. `n_step` must be at
            least 1, in which case we use the standard one-step transition, i.e.
            (s_t, a_t, r_t, d_t, s_t+1, e_t).
          discount: Discount factor to apply. This corresponds to the
            agent's discount in the class docstring.
          net_ids_to_keys: A list of network keys. By indexing this list with the
          network_id the corresponding network key will be returned.
          table_network_config: A dictionary mapping table names to lists of
            network names.
          priority_fns: See docstring for BaseAdder.

        Raises:
          ValueError: If n_step is less than 1.
        """
        # Makes the additional discount a float32, which means that it will be
        # upcast if rewards/discounts are float64 and left alone otherwise.
        self.n_step = n_step
        self._net_ids_to_keys = net_ids_to_keys
        self._discount = tree.map_structure(np.float32, discount)
        self._first_idx = 0
        self._last_idx = 0
        self._table_network_config = table_network_config

        ReverbParallelAdder.__init__(
            self,
            client=client,
            max_sequence_length=n_step + 1,
            priority_fns=priority_fns,
            max_in_flight_items=max_in_flight_items,
            use_next_extras=True,
        )

    def _write(self) -> None:
        # Convenient getters for use in tree operations.
        def get_first(x: np.array) -> np.array:
            return x[self._first_idx]

        def get_last(x: np.array) -> np.array:
            return x[self._last_idx]

        # Note: this getter is meant to be used on a TrajectoryWriter.history to
        # obtain its numpy values.
        def get_all_np(x: np.array) -> np.array:
            return x[self._first_idx : self._last_idx].numpy()

        # Get the state, action, next_state, as well as possibly extras for the
        # transition that is about to be written.
        history = self._writer.history
        s, e, a = tree.map_structure(
            get_first, (history["observations"], history["extras"], history["actions"])
        )

        s_, e_ = tree.map_structure(
            get_last, (history["observations"], history["extras"])
        )

        # # Maybe get extras to add to the transition later.
        # if 'extras' in history:
        #     extras = tree.map_structure(get_first, history['extras'])

        # Note: at the beginning of an episode we will add the initial N-1
        # transitions (of size 1, 2, ...) and at the end of an episode (when
        # called from write_last) we will write the final transitions of size (N,
        # N-1, ...). See the Note in the docstring.
        # Get numpy view of the steps to be fed into the priority functions.

        rewards, discounts = tree.map_structure(
            get_all_np, (history["rewards"], history["discounts"])
        )
        # Compute discounted return and geometric discount over n steps.
        n_step_return, total_discount = self._compute_cumulative_quantities(
            rewards, discounts
        )

        # Append the computed n-step return and total discount.
        # Note: if this call to _write() is within a call to _write_last(), then
        # this is the only data being appended and so it is not a partial append.
        self._writer.append(
            dict(n_step_return=n_step_return, total_discount=total_discount),
            partial_step=self._writer.episode_steps <= self._last_idx,
        )
        # This should be done immediately after self._writer.append so the history
        # includes the recently appended data.
        history = self._writer.history

        # Form the n-step transition by using the following:
        # the first observation and action in the buffer, along with the cumulative
        # reward and discount computed above.
        n_step_return, total_discount = tree.map_structure(
            lambda x: x[-1], (history["n_step_return"], history["total_discount"])
        )
        transition = mava_types.Transition(
            observations=s,
            extras=e,
            actions=a,
            rewards=n_step_return,
            discounts=total_discount,
            next_observations=s_,
            next_extras=e_,
        )

        # Calculate the priority for this transition.
        table_priorities = acme_utils.calculate_priorities(
            self._priority_fns, transition
        )

        # Add the experience to the trainer tables in the correct form.
        self.write_experience_to_tables(transition, table_priorities)

    @classmethod
    def signature(
        cls,
        environment_spec: mava_specs.EnvironmentSpec,
        extras_spec: tf.TypeSpec = {},
    ) -> tf.TypeSpec:
        """Signature for adder.

        Args:
            environment_spec (mava_specs.EnvironmentSpec): MA environment spec.
            extras_spec (tf.TypeSpec, optional): Spec for extras data. Defaults to {}.

        Returns:
            tf.TypeSpec: Signature for transition adder.
        """

        # This function currently assumes that self._discount is a scalar.
        # If it ever becomes a nested structure and/or a np.ndarray, this method
        # will need to know its structure / shape. This is because the signature
        # discount shape is the environment's discount shape and this adder's
        # discount shape broadcasted together. Also, the reward shape is this
        # signature discount shape broadcasted together with the environment
        # reward shape. As long as self._discount is a scalar, it will not affect
        # either the signature discount shape nor the signature reward shape, so we
        # can ignore it.

        agent_specs = environment_spec.get_agent_specs()
        agents = environment_spec.get_agent_ids()
        env_extras_spec = environment_spec.get_extra_specs()
        extras_spec.update(env_extras_spec)

        obs_specs = {}
        act_specs = {}
        reward_specs = {}
        step_discount_specs = {}
        for agent in agents:

            rewards_spec, step_discounts_spec = tree_utils.broadcast_structures(
                agent_specs[agent].rewards, agent_specs[agent].discounts
            )

            rewards_spec = tree.map_structure(
                _broadcast_specs, rewards_spec, step_discounts_spec
            )
            step_discounts_spec = tree.map_structure(copy.deepcopy, step_discounts_spec)

            obs_specs[agent] = agent_specs[agent].observations
            act_specs[agent] = agent_specs[agent].actions
            reward_specs[agent] = rewards_spec
            step_discount_specs[agent] = step_discounts_spec

        transition_spec = types.Transition(
            observations=obs_specs,
            next_observations=obs_specs,
            actions=act_specs,
            rewards=reward_specs,
            discounts=step_discount_specs,
            extras=extras_spec,
            next_extras=extras_spec,
        )

        return tree.map_structure_with_path(
            base.spec_like_to_tensor_spec, transition_spec
        )
