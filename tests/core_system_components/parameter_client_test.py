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

"""Tests for parameter client class for Jax-based Mava systems"""

import copy
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Set, Union

import jax
import numpy as np
import pytest

from mava.callbacks.base import Callback
from mava.systems.parameter_client import ParameterClient
from mava.systems.parameter_server import ParameterServer


class MockParameterServer(ParameterServer):
    def __init__(
        self,
        store: SimpleNamespace,
        components: List[Callback],
        set_parameter_keys: Union[str, Sequence[str]],
    ) -> None:
        """Initialize mock parameter server."""
        self.store = store
        self.callbacks = components
        self.set_parameter_keys = set_parameter_keys

    def _increment_get_parameters(
        self, names: Union[str, Sequence[str], Set[str]]
    ) -> None:
        """Dummy method to update get parameters before updating client"""
        for name in names:
            if name.split("_")[0] == "key" or name.split("_")[0] == "allkey":
                self.store.parameters[name] += 1
            elif "network" in name.split("-")[0]:
                self.store.parameters[name]["layer_0"]["weights"] += 1
                self.store.parameters[name]["layer_0"]["biases"] += 1

    def get_parameters(self, names: Union[str, Sequence[str]]) -> Any:
        """Dummy method for returning get parameters"""
        self.store._param_names = names

        # Manually increment all parameters except the set parameters
        # and add them to store to simulate parameters that have changed.
        get_names = set(names) - set(self.set_parameter_keys)
        self._increment_get_parameters(names=get_names)
        get_params = {name: self.store.parameters[name] for name in names}
        self.store.get_parameters = get_params

        return self.store.get_parameters

    def set_parameters(self, set_params: Dict[str, Any]) -> None:
        """Overwrite set parameters method"""

        self.store._set_params = set_params

        for key in set_params:
            self.store.parameters[key] = copy.deepcopy(set_params[key])


def increment_set_parameters(
    params: Dict[str, Any], names: Union[str, Sequence[str], Set[str]]
) -> None:
    """Utility function for incrementing parameter client parameters"""
    for key in names:
        if key.split("_")[0] == "key" or key.split("_")[0] == "allkey":
            params[key] += 1
        elif "network" in key.split("-")[0]:
            params[key]["layer_0"]["weights"] += 1
            params[key]["layer_0"]["biases"] += 1


@pytest.fixture
def mock_parameter_server() -> ParameterServer:
    """Create mock parameter server"""
    param_server = MockParameterServer(
        store=SimpleNamespace(
            parameters={
                "key_0": np.array(0, dtype=np.int32),
                "key_1": np.array(1, dtype=np.float32),
                "key_2": np.array(2, dtype=np.int32),
                "key_3": np.array(3, dtype=np.int32),
                "key_4": np.array(4, dtype=np.int32),
                "policy_network-network_key_0": {
                    "layer_0": {"weights": 0, "biases": 0}
                },
                "critic_network-network_key_1": {
                    "layer_0": {"weights": 1, "biases": 1}
                },
                "allkey_0": np.array(0, dtype=np.int32),
            },
        ),
        components=[],
        set_parameter_keys=["key_0", "key_2"],
    )

    return param_server


@pytest.fixture()
def parameter_client(mock_parameter_server: ParameterServer) -> ParameterClient:
    """Creates a mock parameter client for testing

    Args:
        mock_parameter_server: ParameterServer

    Returns:
        A parameter client object.
    """

    param_client = ParameterClient(
        server=mock_parameter_server,
        parameters={
            "key_0": np.array(0, dtype=np.int32),
            "key_1": np.array(1, dtype=np.float32),
            "key_2": np.array(2, dtype=np.int32),
            "key_3": np.array(3, dtype=np.int32),
            "key_4": np.array(4, dtype=np.int32),
            "policy_network-network_key_0": {"layer_0": {"weights": 0, "biases": 0}},
            "critic_network-network_key_1": {"layer_0": {"weights": 1, "biases": 1}},
            "allkey_0": np.array(0, dtype=np.int32),
        },
        # must force this to be false, otherwise would need to create lp node for server
        multi_process=False,
        get_keys=[
            "key_0",
            "key_1",
            "key_2",
            "key_3",
            "key_4",
            "policy_network-network_key_0",
            "critic_network-network_key_1",
        ],
        set_keys=[
            "key_0",
            "key_2",
        ],
        update_period=10,
    )

    return param_client


def test_add_and_wait(parameter_client: ParameterClient) -> None:
    """Test add and wait method."""
    parameter_client.add_and_wait(params={"new_key": "new_value"})

    assert parameter_client._server.store._add_to_params == {"new_key": "new_value"}


def test_get_and_wait(parameter_client: ParameterClient) -> None:
    """Test get and wait method."""
    parameter_client.get_and_wait()
    # check that all get parameters have been incremented and updated
    # except for the set parameters
    assert parameter_client._parameters == {
        "key_0": np.array(0, dtype=np.int32),
        "key_1": np.array(2, dtype=np.float32),
        "key_2": np.array(2, dtype=np.int32),
        "key_3": np.array(4, dtype=np.int32),
        "key_4": np.array(5, dtype=np.int32),
        "policy_network-network_key_0": {"layer_0": {"weights": 1, "biases": 1}},
        "critic_network-network_key_1": {"layer_0": {"weights": 2, "biases": 2}},
        "allkey_0": np.array(0, dtype=np.int32),
    }


def test_get_all_and_wait(parameter_client: ParameterClient) -> None:
    """Test get all and wait method."""
    parameter_client.get_all_and_wait()
    # check that all parameters have been incremented and updated
    # except for the set parameters
    assert parameter_client._parameters == {
        "key_0": np.array(0, dtype=np.int32),
        "key_1": np.array(2, dtype=np.float32),
        "key_2": np.array(2, dtype=np.int32),
        "key_3": np.array(4, dtype=np.int32),
        "key_4": np.array(5, dtype=np.int32),
        "policy_network-network_key_0": {"layer_0": {"weights": 1, "biases": 1}},
        "critic_network-network_key_1": {"layer_0": {"weights": 2, "biases": 2}},
        "allkey_0": np.array(1, dtype=np.int32),
    }


def test_set_and_wait(parameter_client: ParameterClient) -> None:
    """Test set and wait method."""
    # should set parameters from client to server.
    increment_set_parameters(
        params=parameter_client._parameters, names=parameter_client._set_keys
    )

    parameter_client.set_and_wait()

    assert parameter_client._server.store._set_params == {
        "key_0": np.array(1, dtype=np.int32),
        "key_2": np.array(3, dtype=np.int32),
    }

    assert parameter_client._server.store.parameters == {
        "key_0": np.array(1, dtype=np.int32),
        "key_1": np.array(1, dtype=np.float32),
        "key_2": np.array(3, dtype=np.int32),
        "key_3": np.array(3, dtype=np.int32),
        "key_4": np.array(4, dtype=np.int32),
        "policy_network-network_key_0": {"layer_0": {"weights": 0, "biases": 0}},
        "critic_network-network_key_1": {"layer_0": {"weights": 1, "biases": 1}},
        "allkey_0": np.array(0, dtype=np.int32),
    }


def test_get_async(parameter_client: ParameterClient) -> None:
    """Test get async method"""

    # test that get_call_counter is incremented
    parameter_client.get_async()
    assert parameter_client._get_call_counter == 1

    # set call counter equal to update period and verify the request was made
    parameter_client._get_call_counter = 10
    parameter_client.get_async()

    assert parameter_client._parameters == {
        "key_0": np.array(0, dtype=np.int32),
        "key_1": np.array(2, dtype=np.float32),
        "key_2": np.array(2, dtype=np.int32),
        "key_3": np.array(4, dtype=np.int32),
        "key_4": np.array(5, dtype=np.int32),
        "policy_network-network_key_0": {"layer_0": {"weights": 1, "biases": 1}},
        "critic_network-network_key_1": {"layer_0": {"weights": 2, "biases": 2}},
        "allkey_0": np.array(0, dtype=np.int32),
    }
    assert parameter_client._get_call_counter == 0
    assert parameter_client._get_future is None


def test_set_async(parameter_client: ParameterClient) -> None:
    """Set set async method"""

    # test that set_call_counter is incremented
    parameter_client.set_async()
    assert parameter_client._set_call_counter == 1

    # set call counter equal to update period and verify the request was made
    parameter_client._set_call_counter = 10

    # increment set parameters, call set_async and verify that
    # parameters were set asynchronously
    increment_set_parameters(
        params=parameter_client._parameters, names=parameter_client._set_keys
    )

    parameter_client.set_async()

    assert parameter_client._server.store._set_params == {
        "key_0": np.array(1, dtype=np.int32),
        "key_2": np.array(3, dtype=np.int32),
    }
    assert parameter_client._set_call_counter == 0

    # assert that _set_future is None again after request was made
    parameter_client.set_async()
    assert parameter_client._set_future is None


def test_set_and_get_async(parameter_client: ParameterClient) -> None:
    """Set set async method"""

    # test that set_and_get_call_counter is incremented
    parameter_client.set_and_get_async()
    assert parameter_client._set_get_call_counter == 1

    # set counter equal to update period and verify the request was made
    parameter_client._set_get_call_counter = 10

    # increment set parameters
    increment_set_parameters(
        params=parameter_client._parameters, names=parameter_client._set_keys
    )

    parameter_client.set_and_get_async()
    assert parameter_client._parameters == {
        "key_0": np.array(1, dtype=np.int32),
        "key_1": np.array(2, dtype=np.float32),
        "key_2": np.array(3, dtype=np.int32),
        "key_3": np.array(4, dtype=np.int32),
        "key_4": np.array(5, dtype=np.int32),
        "policy_network-network_key_0": {"layer_0": {"weights": 1, "biases": 1}},
        "critic_network-network_key_1": {"layer_0": {"weights": 2, "biases": 2}},
        "allkey_0": np.array(0, dtype=np.int32),
    }

    assert parameter_client._set_get_call_counter == 0

    parameter_client.set_and_get_async()

    assert parameter_client._set_get_future is None


def test_add_async(parameter_client: ParameterClient) -> None:
    """Test add async method."""

    assert parameter_client._add_future is None

    parameter_client.add_async(params={"new_key": "new_value"})
    assert parameter_client._add_future is not None
    assert parameter_client._server.store._add_to_params == {"new_key": "new_value"}
    assert parameter_client._async_add_buffer == {}
    assert parameter_client._add_future.done()

    # force future to not be done
    # assert that new parameter is added to async buffer
    parameter_client._add_future.done = lambda: False
    parameter_client._async_add_buffer = {"new_key_1": 1}
    parameter_client.add_async(params={"new_key_1": 1})

    assert parameter_client._async_add_buffer == {"new_key_1": 2}

    # force future to not be done
    # assert that new parameter is added to async buffer
    parameter_client._add_future.done = lambda: False
    parameter_client._async_add_buffer = {}
    parameter_client.add_async(params={"new_key_2": 1})

    assert parameter_client._async_add_buffer == {"new_key_2": 1}


def test__copy(parameter_client: ParameterClient) -> None:
    """Test _copy method with different kinds of new parameters"""
    parameter_client._copy(
        new_parameters={
            "policy_network-network_key_0": {
                "layer_0": {"weights": "new_weights", "biases": "new_biases"}
            },
            "key_2": np.array(20, dtype=np.int32),
            "key_4": np.array([40], dtype=np.int32),
        }
    )

    assert parameter_client._parameters["policy_network-network_key_0"] == {
        "layer_0": {"weights": "new_weights", "biases": "new_biases"}
    }
    assert parameter_client._parameters["key_2"] == 20
    assert parameter_client._parameters["key_4"] == 40


def test__copy_not_implemented_error(parameter_client: ParameterClient) -> None:
    """Test that NotImplementedError is raised when a new parameter of the wrong \
        type is passed in."""

    with pytest.raises(NotImplementedError):
        parameter_client._copy(
            new_parameters={"wrong_type_parameter": lambda: "wrong_type"}
        )


def test__copy_device_not_implemented_error(parameter_client: ParameterClient) -> None:
    """Test that NotImplementedError is raised when a new parameter of the wrong \
        type is passed in."""

    parameter_client._devices = {"device_1": "dummy_device"}

    with pytest.raises(NotImplementedError):
        parameter_client._copy(
            new_parameters={
                "policy_network-network_key_0": {
                    "layer_0": {"weights": "new_weights", "biases": "new_biases"}
                }
            },
        )


def test_copy_array_with_device(parameter_client: ParameterClient) -> None:
    """Test that new parameters are set on a device when a device \
        is give."""
    local_devices = jax.local_devices()
    parameter_client._devices = {"key_1": local_devices[0]}

    parameter_client._copy(new_parameters={"key_1": np.array([0])})

    assert type(parameter_client._parameters["key_1"]).__name__ == "DeviceArray"
    assert parameter_client._parameters["key_1"] == jax.numpy.array([0])


def test_copy_tuple_with_device(parameter_client: ParameterClient) -> None:
    """Test that new parameters are set on a device when a device \
        is given."""
    parameter_client._parameters.update({"tuple_key_0": [1, 2]})
    local_devices = jax.local_devices()
    parameter_client._devices = {
        "tuple_key_0": (local_devices[0], local_devices[0])  # type: ignore
    }

    parameter_client._copy(
        new_parameters={"tuple_key_0": (np.array([11]), np.array([22]))}
    )

    assert (
        type(parameter_client._parameters["tuple_key_0"][0]).__name__ == "DeviceArray"
    )
    assert (
        type(parameter_client._parameters["tuple_key_0"][1]).__name__ == "DeviceArray"
    )
    assert parameter_client._parameters["tuple_key_0"] == [
        jax.numpy.array([11]),
        jax.numpy.array([22]),
    ]
