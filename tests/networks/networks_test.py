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

"""Network tests."""
from typing import Any, Dict

import numpy as np
import pytest
import sonnet as snt
import tensorflow as tf
from absl import flags

from mava import specs as mava_specs
from mava.components.tf.architectures.decentralised import (
    DecentralisedQValueActorCritic,
    DecentralisedValueActor,
    DecentralisedValueActorCritic,
)
from mava.components.tf.networks.continuous import LayerNormAndResidualMLP, LayerNormMLP
from mava.systems.tf import dial, mad4pg, maddpg, madqn, mappo, qmix, vdn
from mava.utils.environments import debugging_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "discrete",
    "Environment action space type (str).",
)


@pytest.mark.parametrize(
    "system",
    [maddpg, mad4pg, madqn, mappo, qmix, vdn, dial],
)
class TestNetworkAgentKeys:
    """Test that we get the correct agent networks from make network functions."""

    def test_shared_networks(self, system: Any) -> None:
        """Test shared networks from make networks function.

        Args:
            system (Any): A Mava system.
        """
        # Environment.
        env = debugging_utils.make_environment(
            env_name=FLAGS.env_name, action_space=FLAGS.action_space, evaluation=False
        )
        environment_spec = mava_specs.MAEnvironmentSpec(env)

        # Test shared weights setup.
        agent_net_keys = {
            "agent_0": "agent",
            "agent_1": "agent",
            "agent_2": "agent",
        }

        params = dict(
            environment_spec=environment_spec,
            agent_net_keys=agent_net_keys,
            vmin=-150 if system == mad4pg else None,
            vmax=150 if system == mad4pg else None,
        )

        # Filter out None params
        params = {k: v for k, v in params.items() if v is not None}

        # Than pass params
        networks = system.make_default_networks(**params)

        for key in networks.keys():
            assert (
                "agent" == list(networks[key].keys())[0]
            ), "Incorrect agent networks returned."
            assert (
                len(networks[key].keys()) == 1
            ), "Incorrect number of networks returned."

    def test_non_shared_networks(self, system: Any) -> None:
        """Test non-shared networks from make networks function.

        Args:
            system (Any): A Mava system.
        """
        # Environment.
        env = debugging_utils.make_environment(
            env_name=FLAGS.env_name, action_space=FLAGS.action_space, evaluation=False
        )
        environment_spec = mava_specs.MAEnvironmentSpec(env)

        # Test individual agent network setup.
        agent_net_keys = {
            "agent_0": "agent_0",
            "agent_1": "agent_1",
            "agent_2": "agent_2",
        }

        if system == mad4pg:
            networks = system.make_default_networks(  # type: ignore
                environment_spec=environment_spec,  # type: ignore
                agent_net_keys=agent_net_keys,
                vmin=-150,
                vmax=150,
            )
        else:
            networks = system.make_default_networks(  # type: ignore
                environment_spec=environment_spec,  # type: ignore
                agent_net_keys=agent_net_keys,
            )
        for key in networks.keys():
            assert ["agent_0", "agent_1", "agent_2"] == list(
                networks[key].keys()
            ), "Incorrect agent networks returned."
            assert len(networks[key].keys()) == len(
                agent_net_keys.keys()
            ), "Incorrect number of networks returned."

    def test_network_seed_is_passed(self, system: Any) -> None:
        """Test passing of network seed.

        Args:
            system (Any): system.
        """
        test_seed = 42

        # Environment.
        env = debugging_utils.make_environment(
            env_name=FLAGS.env_name, action_space=FLAGS.action_space, evaluation=False
        )
        environment_spec = mava_specs.MAEnvironmentSpec(env)

        # Test shared weights setup.
        agent_net_keys = {
            "agent_0": "agent",
            "agent_1": "agent",
            "agent_2": "agent",
        }

        if system == mad4pg:
            networks = system.make_default_networks(
                environment_spec=environment_spec,
                agent_net_keys=agent_net_keys,
                vmin=-150,
                vmax=150,
                seed=test_seed,
            )
        else:
            networks = system.make_default_networks(
                environment_spec=environment_spec,
                agent_net_keys=agent_net_keys,
                seed=test_seed,
            )

        for key in networks:
            for agent in networks[key]:
                if type(networks[key][agent]) == snt.Sequential:
                    for layer in networks[key][agent]._layers:
                        if hasattr(layer, "_seed"):
                            assert layer._seed == test_seed


@pytest.mark.parametrize(
    "architecture",
    [
        dict(
            network_mapping={
                "value_networks": "q_networks",
            },
            architecture=DecentralisedValueActor,
            system=madqn,
        ),
        dict(
            network_mapping={
                "observation_networks": "observations",
                "policy_networks": "policies",
                "critic_networks": "critics",
            },
            architecture=DecentralisedQValueActorCritic,
            system=maddpg,
        ),
        dict(
            network_mapping={
                "observation_networks": "observations",
                "policy_networks": "policies",
                "critic_networks": "critics",
            },
            architecture=DecentralisedValueActorCritic,
            system=mappo,
        ),
    ],
)
class TestArchitectureAgentKeys:
    """Test that we get the correct agent networks from the architectures."""

    def test_shared_networks(self, architecture: Dict) -> None:
        """Test that shared weight architectures work.

        Args:
            architecture (Dict): Dict containing a network mapping (from return networks
                to arch params), architecture and system.
        """
        # Environment.
        env = debugging_utils.make_environment(
            env_name=FLAGS.env_name, action_space=FLAGS.action_space, evaluation=False
        )
        environment_spec = mava_specs.MAEnvironmentSpec(env)

        # Test shared weights setup.
        agent_net_keys = {
            "agent_0": "agent",
            "agent_1": "agent",
            "agent_2": "agent",
        }
        networks = architecture["system"].make_default_networks(  # type: ignore
            environment_spec=environment_spec,  # type: ignore
            agent_net_keys=agent_net_keys,
        )

        for key in networks.keys():
            assert (
                "agent" == list(networks[key].keys())[0]
            ), "Incorrect agent networks returned."
            assert (
                len(networks[key].keys()) == 1
            ), "Incorrect number of networks returned."

        env_config = {
            "environment_spec": environment_spec,
            "agent_net_keys": agent_net_keys,
        }

        # Add network arch config
        network_config = {}
        for key, value in architecture["network_mapping"].items():
            network_config[key] = networks[value]

        architecture_config = dict(env_config, **network_config)

        system_networks = architecture["architecture"](
            **architecture_config
        ).create_system()
        for key in system_networks.keys():
            assert (
                "agent" == list(system_networks[key].keys())[0]
            ), "Incorrect agent networks returned."
            assert (
                len(system_networks[key].keys()) == 1
            ), "Incorrect number of networks returned."

    def test_non_shared_networks(self, architecture: Dict) -> None:
        """Test that non shared weight architectures work.

        Args:
            architecture (Dict): Dict containing a network mapping (from return networks
                to arch params), architecture and system.
        """
        # Environment.
        env = debugging_utils.make_environment(
            env_name=FLAGS.env_name, action_space=FLAGS.action_space, evaluation=False
        )
        environment_spec = mava_specs.MAEnvironmentSpec(env)

        # Test individual agent network setup.
        agent_net_keys = {
            "agent_0": "agent_0",
            "agent_1": "agent_1",
            "agent_2": "agent_2",
        }
        networks = architecture["system"].make_default_networks(  # type: ignore
            environment_spec=environment_spec,  # type: ignore
            agent_net_keys=agent_net_keys,
        )
        for key in networks.keys():
            assert ["agent_0", "agent_1", "agent_2"] == list(
                networks[key].keys()
            ), "Incorrect agent networks returned."
            assert len(networks[key].keys()) == len(
                agent_net_keys.keys()
            ), "Incorrect number of networks returned."


@pytest.mark.parametrize(
    "network",
    [
        (LayerNormMLP, {"layer_sizes": [10, 10]}),
        (LayerNormAndResidualMLP, {"hidden_size": 10, "num_blocks": 1}),
    ],
)
class TestNetworksSeeding:
    def test_network_reproducibility_0_same_seed(self, network: Any) -> None:
        """Test with same seed, networks are the same.

        Args:
            system (Any): network.
        """
        test_seed = 42
        network, network_params = network

        # Network 1
        net = network(**network_params, seed=test_seed)
        # Simple forward pass
        net(tf.ones([10, 10]))

        # Network 2
        net2 = network(**network_params, seed=test_seed)
        # Simple forward pass
        net2(tf.ones([10, 10]))

        np.testing.assert_array_equal(
            np.concatenate([x.numpy().flatten() for x in net.variables]),
            np.concatenate([x.numpy().flatten() for x in net2.variables]),
        )

    def test_network_reproducibility_1_no_seed(self, network: Any) -> None:
        """Test with no seed, networks are different.

        Args:
            system (Any): network.
        """
        network, network_params = network

        # Network 1
        net = network(**network_params)
        # Simple forward pass
        net(tf.ones([10, 10]))

        # Network 2
        net2 = network(**network_params)
        # Simple forward pass
        net2(tf.ones([10, 10]))

        assert not np.array_equal(
            np.concatenate([x.numpy().flatten() for x in net.variables]),
            np.concatenate([x.numpy().flatten() for x in net2.variables]),
        )

    def test_network_reproducibility_2_diff_seed(self, network: Any) -> None:
        """Test with diff seeds, networks are different.

        Args:
            system (Any): network.
        """
        network, network_params = network
        test_seed = 42
        test_seed2 = 43

        # Network 1
        net = network(**network_params, seed=test_seed)
        # Simple forward pass
        net(tf.ones([10, 10]))

        # Network 2
        net2 = network(**network_params, seed=test_seed2)
        # Simple forward pass
        net2(tf.ones([10, 10]))

        assert not np.array_equal(
            np.concatenate([x.numpy().flatten() for x in net.variables]),
            np.concatenate([x.numpy().flatten() for x in net2.variables]),
        )
