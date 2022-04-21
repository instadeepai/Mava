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

"""Tests for config class for Jax-based Mava systems"""

from typing import Any, Dict

import pytest
from reverb import item_selectors, pybind, reverb_types

from mava.testing.building.mocks import return_test_system
from mava.testing.utils import assert_if_value_is_not_none

from .data_server_test_data import transition_adder_data_server_test_cases


@pytest.mark.parametrize(
    "components",
    transition_adder_data_server_test_cases,
)
class TestDataServer:
    # Adapted from
    # https://github.com/deepmind/reverb/blob/c5ea7c37118d0de4ff2320bf5519ba79ad1d4284/reverb/server_test.py#L102
    def _check_selector_proto(
        self, expected_selector: reverb_types.SelectorType, proto_msg: Any
    ) -> None:
        if isinstance(expected_selector, item_selectors.Uniform):
            assert proto_msg.HasField("uniform")
        elif isinstance(expected_selector, item_selectors.Prioritized):
            assert proto_msg.HasField("prioritized")
        elif isinstance(expected_selector, pybind.HeapSelector):
            assert proto_msg.HasField("heap")
        elif isinstance(expected_selector, item_selectors.Fifo):
            assert proto_msg.HasField("fifo")
        elif isinstance(expected_selector, item_selectors.Lifo):
            assert proto_msg.HasField("lifo")
        else:
            raise ValueError(f"Unknown selector: {expected_selector}")

    def test_data_server(
        self,
        components: Dict[str, Dict],
    ) -> None:
        """Test if system builder instantiates processes as expected."""
        # Get specified component
        test_system = return_test_system(components["component"])
        system_config = components["system_config"]
        test_system.build(**system_config)
        test_system._builder.data_server()

        # Assuming a single table for now
        table = test_system._builder.store.data_tables[0]
        table_info = table.info

        # Testing table is set correctly
        assert (
            table_info.name
            == f"{system_config['data_server_name']}_{list(test_system._builder.store.table_network_config.keys())[0]}"  # noqa:E501
        )

        # Off policy params
        assert_if_value_is_not_none(system_config.get("max_size"), table_info.max_size)
        assert_if_value_is_not_none(
            system_config.get("max_times_sampled"), table_info.max_times_sampled
        )
        assert_if_value_is_not_none(
            system_config.get("min_data_server_size"),
            table_info.rate_limiter_info.min_size_to_sample,
        )
        if system_config.get("sampler"):
            self._check_selector_proto(
                system_config["sampler"], table_info.sampler_options
            )
        if system_config.get("remover"):
            self._check_selector_proto(
                system_config["remover"], table_info.remover_options
            )

        # On policy params
        assert_if_value_is_not_none(
            system_config.get("max_queue_size"), table_info.max_size
        )
