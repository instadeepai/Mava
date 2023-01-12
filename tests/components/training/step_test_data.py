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

"""Dummy data for Step unit test"""

from typing import Any, Tuple

import jax.numpy as jnp
import reverb
from jax.tree_util import register_pytree_node_class

from mava.types import OLT


@register_pytree_node_class
class MockStep:
    """Mock Step for data sampling"""

    def __init__(
        self,
        observations: Any,
        actions: Any,
        rewards: Any,
        discounts: Any,
        start_of_episode: Any,
        extras: Any,
        next_extras: Any,
    ) -> None:
        """Init"""
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.discounts = discounts
        self.start_of_episode = start_of_episode
        self.extras = extras
        self.next_extras = next_extras

    def tree_flatten(self) -> Tuple:
        """Needed for the pytree"""
        children = (
            self.observations,
            self.actions,
            self.rewards,
            self.discounts,
            self.start_of_episode,
            self.extras,
            self.next_extras,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        """Needed for the pytree"""
        return cls(*children)


dummy_sample = reverb.ReplaySample(
    info=reverb.SampleInfo(
        key=jnp.array([12865463882669798571, 16233754366579778025], dtype=jnp.uint64),
        probability=jnp.array([1.0, 1.0]),
        table_size=jnp.array([1, 1]),
        priority=jnp.array([1.0, 1.0]),
    ),
    data=MockStep(
        observations={
            "agent_0": OLT(
                observation=jnp.array(
                    [
                        [
                            [
                                0.0,
                                0.0,
                                0.8539885,
                                -0.518841,
                                0.0,
                                -1.1304581,
                                0.13541625,
                                -0.8415354,
                                0.8974394,
                                -0.53403145,
                                0.541699,
                                0.030427,
                                0.5862218,
                                -1.7108788,
                                0.96656066,
                            ],
                            [
                                0.0,
                                -0.0,
                                0.8539885,
                                -0.518841,
                                0.02,
                                -1.1304581,
                                0.13541625,
                                -0.8415354,
                                0.84743947,
                                -0.58403146,
                                0.541699,
                                0.030427,
                                0.5862218,
                                -1.7108788,
                                0.96656066,
                            ],
                            [
                                0.0,
                                -0.0,
                                0.8539885,
                                -0.518841,
                                0.04,
                                -1.1304581,
                                0.13541625,
                                -0.7915354,
                                0.80993944,
                                -0.6215314,
                                0.541699,
                                0.030427,
                                0.5862218,
                                -1.7108788,
                                0.96656066,
                            ],
                        ],
                        [
                            [
                                0.11865234,
                                0.65267944,
                                1.0183928,
                                -0.11464486,
                                0.2,
                                -1.2948624,
                                -0.2687799,
                                -0.9584569,
                                0.4132104,
                                -0.3758324,
                                -0.09247538,
                                -0.1339773,
                                0.18202564,
                                -1.8752831,
                                0.5623645,
                            ],
                            [
                                0.08898926,
                                -0.01049042,
                                1.0272918,
                                -0.1156939,
                                0.22,
                                -1.3037614,
                                -0.26773086,
                                -1.0292267,
                                0.4842679,
                                -0.36538208,
                                -0.13393201,
                                -0.14287622,
                                0.18307468,
                                -1.884182,
                                0.56341356,
                            ],
                            [
                                0.06674194,
                                0.4921322,
                                1.033966,
                                -0.06648068,
                                0.24,
                                -1.3104355,
                                -0.3169441,
                                -1.0823039,
                                0.487561,
                                -0.30754435,
                                -0.21502449,
                                -0.14955041,
                                0.13386145,
                                -1.8908563,
                                0.5142003,
                            ],
                        ],
                    ],
                    dtype=jnp.float32,
                ),
                legal_actions=jnp.array(
                    [
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                    ]
                ),
            ),
            "agent_1": OLT(
                observation=jnp.array(
                    [
                        [
                            [
                                0.0,
                                0.0,
                                0.01245314,
                                0.37859842,
                                0.0,
                                0.87196237,
                                -0.31121764,
                                0.8415354,
                                -0.8974394,
                                0.30750394,
                                -0.35574046,
                                -0.28892276,
                                -0.7620232,
                                -0.86934346,
                                0.06912122,
                            ],
                            [
                                -0.0,
                                -0.5,
                                0.01245314,
                                0.3285984,
                                0.02,
                                0.87196237,
                                -0.26121765,
                                0.8415354,
                                -0.84743947,
                                0.25750396,
                                -0.30574045,
                                -0.28892276,
                                -0.7120232,
                                -0.86934346,
                                0.11912122,
                            ],
                            [
                                0.5,
                                -0.375,
                                0.06245314,
                                0.29109842,
                                0.04,
                                0.82196236,
                                -0.22371764,
                                0.7915354,
                                -0.80993944,
                                0.17000394,
                                -0.26824045,
                                -0.33892277,
                                -0.6745232,
                                -0.9193434,
                                0.15662122,
                            ],
                        ],
                        [
                            [
                                -0.82494366,
                                0.26677936,
                                0.05993592,
                                0.29856554,
                                0.2,
                                0.8244796,
                                -0.23118475,
                                0.9584569,
                                -0.4132104,
                                0.5826245,
                                -0.50568575,
                                -0.33640555,
                                -0.68199027,
                                -0.9168262,
                                0.14915411,
                            ],
                            [
                                -0.6187078,
                                0.7000845,
                                -0.00193486,
                                0.368574,
                                0.22,
                                0.8863504,
                                -0.3011932,
                                1.0292267,
                                -0.4842679,
                                0.6638445,
                                -0.6181999,
                                -0.27453476,
                                -0.7519987,
                                -0.85495543,
                                0.07914566,
                            ],
                            [
                                -0.46403083,
                                0.5250634,
                                -0.04833794,
                                0.42108032,
                                0.24,
                                0.93275344,
                                -0.35369954,
                                1.0823039,
                                -0.487561,
                                0.77475953,
                                -0.70258546,
                                -0.22813168,
                                -0.8045051,
                                -0.8085523,
                                0.02663932,
                            ],
                        ],
                    ],
                    dtype=jnp.float32,
                ),
                legal_actions=jnp.array(
                    [
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                    ]
                ),
            ),
            "agent_2": OLT(
                observation=jnp.array(
                    [
                        [
                            [
                                0.0,
                                0.0,
                                0.31995708,
                                0.02285798,
                                0.0,
                                -1.1768473,
                                0.42486167,
                                -0.30750394,
                                0.35574046,
                                0.53403145,
                                -0.541699,
                                0.56445843,
                                0.0445228,
                                -0.5964267,
                                -0.40628275,
                            ],
                            [
                                -0.5,
                                -0.0,
                                0.2699571,
                                0.02285798,
                                0.02,
                                -1.1268474,
                                0.42486167,
                                -0.25750396,
                                0.30574045,
                                0.58403146,
                                -0.541699,
                                0.61445844,
                                0.0445228,
                                -0.5464267,
                                -0.40628275,
                            ],
                            [
                                -0.375,
                                -0.0,
                                0.23245709,
                                0.02285798,
                                0.04,
                                -1.0893474,
                                0.42486167,
                                -0.17000394,
                                0.26824045,
                                0.6215314,
                                -0.541699,
                                0.65195847,
                                0.0445228,
                                -0.5089267,
                                -0.40628275,
                            ],
                        ],
                        [
                            [
                                0.25799003,
                                -0.56674236,
                                0.6425604,
                                -0.20712024,
                                0.2,
                                -1.4994507,
                                0.6548399,
                                -0.5826245,
                                0.50568575,
                                0.3758324,
                                0.09247538,
                                0.24185511,
                                0.274501,
                                -0.91903,
                                -0.17630453,
                            ],
                            [
                                0.19349252,
                                -0.42505676,
                                0.66190964,
                                -0.2496259,
                                0.22,
                                -1.5187999,
                                0.69734555,
                                -0.6638445,
                                0.6181999,
                                0.36538208,
                                0.13393201,
                                0.22250587,
                                0.31700668,
                                -0.9383793,
                                -0.13379885,
                            ],
                            [
                                0.64511937,
                                -0.31879258,
                                0.7264216,
                                -0.28150517,
                                0.24,
                                -1.5833119,
                                0.7292248,
                                -0.77475953,
                                0.70258546,
                                0.30754435,
                                0.21502449,
                                0.15799393,
                                0.34888595,
                                -1.0028912,
                                -0.1019196,
                            ],
                        ],
                    ],
                    dtype=jnp.float32,
                ),
                legal_actions=jnp.array(
                    [
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                    ]
                ),
            ),
        },
        actions={
            "agent_0": jnp.array([[3, 2, 0], [4, 0, 3]]),
            "agent_1": jnp.array([[1, 0, 3], [0, 2, 0]]),
            "agent_2": jnp.array([[0, 0, 4], [3, 4, 2]]),
        },
        rewards={
            "agent_0": jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32),
            "agent_1": jnp.array(
                [
                    [0.0897511, 0.14813632, 0.19152136],
                    [0.06387269, 0.00243677, 0.0],
                ],
                dtype=jnp.float32,
            ),
            "agent_2": jnp.array(
                [[0.0, 0.0, 0.0], [0.0897511, 0.14813632, 0.19152136]],
                dtype=jnp.float32,
            ),
        },
        discounts={
            "agent_0": jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32),
            "agent_1": jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32),
            "agent_2": jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32),
        },
        start_of_episode=jnp.array([[True, False, False], [False, False, False]]),
        extras={
            "network_keys": {
                "agent_0": jnp.array([[0, 0, 0], [0, 0, 0]], dtype=jnp.int32),
                "agent_1": jnp.array([[0, 0, 0], [0, 0, 0]], dtype=jnp.int32),
                "agent_2": jnp.array([[0, 0, 0], [0, 0, 0]], dtype=jnp.int32),
            },
            "policy_info": {
                "agent_0": jnp.array(
                    [
                        [-1.5010276, -1.5574824, -1.7098966],
                        [-1.6839617, -1.8447837, -1.4597069],
                    ],
                    dtype=jnp.float32,
                ),
                "agent_1": jnp.array(
                    [
                        [-1.6333038, -1.7833046, -1.4482918],
                        [-1.6957064, -1.4800832, -1.66526],
                    ],
                    dtype=jnp.float32,
                ),
                "agent_2": jnp.array(
                    [
                        [-1.4521754, -1.4560769, -1.8592778],
                        [-1.4220893, -1.78906, -1.5569873],
                    ],
                    dtype=jnp.float32,
                ),
            },
        },
        next_extras={},
    ),
)
