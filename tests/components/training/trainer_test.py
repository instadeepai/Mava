# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Trainer unit test"""

from types import SimpleNamespace
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mava import constants
from mava.callbacks.base import Callback
from mava.components.training.trainer import (
    CustomTrainerInit,
    CustomTrainerInitConfig,
    OneTrainerPerNetworkInit,
    SingleTrainerInit,
)
from mava.systems.builder import Builder
from mava.systems.trainer import Trainer

#######################
# TrainerInit Fixtures
#######################


@pytest.fixture
def single_trainer_init() -> SingleTrainerInit:
    """Creates single trainer init component."""
    return SingleTrainerInit()


@pytest.fixture
def one_trainer_per_network_init() -> OneTrainerPerNetworkInit:
    """Create one trainer per network init component"""
    return OneTrainerPerNetworkInit()


#################
# Mock builders
#################


class MockBuilder(Builder):
    """Mock for the builder"""

    def __init__(
        self,
        components: List[Callback],
        global_config: SimpleNamespace = SimpleNamespace(),
    ) -> None:
        """Creates a mock builder for testing

        Args:
            components : list of mava components
            global_config : global system config
        """
        super().__init__(components, global_config)
        self.store.agents = ["agent_0", "agent_1", "agent_2"]
        self.store.network_factory = lambda: {  # network factory to network
            "network_agent_0": SimpleNamespace(
                policy_params={"weights": 0, "biases": 0},
                critic_params={"weights": 0, "biases": 0},
            ),
            "network_agent_1": SimpleNamespace(
                policy_params={"weights": 1, "biases": 1},
                critic_params={"weights": 1, "biases": 1},
            ),
            "network_agent_2": SimpleNamespace(
                policy_params={"weights": 2, "biases": 2},
                critic_params={"weights": 2, "biases": 2},
            ),
        }

        self.store.policy_optimiser = optax.chain(
            optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
        )
        self.store.critic_optimiser = optax.chain(
            optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
        )

        # Added this to handle the observation normalisation parameters
        obs_spec = SimpleNamespace(
            observations=SimpleNamespace(observation=np.zeros(1)),
        )
        self.store.ma_environment_spec = SimpleNamespace(
            _agent_environment_specs={
                "agent_0": obs_spec,
                "agent_1": obs_spec,
                "agent_2": obs_spec,
            }
        )


@pytest.fixture
def mock_builder_shared_weights_fixed_sampling() -> Builder:
    """Creates mock builder.

    Has shared weights and fixed agent trainer sampling setup.
    """

    mock_builder = MockBuilder(
        global_config=SimpleNamespace(normalize_observations=True), components=[]
    )

    mock_builder.store.unique_net_keys = ["network_agent"]
    mock_builder.store.network_sampling_setup = [
        ["network_agent", "network_agent", "network_agent"]
    ]

    return mock_builder


@pytest.fixture
def mock_builder_no_shared_weights_fixed_sampling() -> Builder:
    """Create mock builder.

    Doesn't shared weights and has fixed agent trainer sampling setup.
    """

    mock_builder = MockBuilder(
        global_config=SimpleNamespace(normalize_observations=True), components=[]
    )

    mock_builder.store.unique_net_keys = [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    mock_builder.store.network_sampling_setup = [
        ["network_agent_0", "network_agent_1", "network_agent_2"]
    ]

    return mock_builder


@pytest.fixture
def mock_builder_no_shared_weights_random_sampling() -> Builder:
    """Create mock builder.

    Doesn't have shared weights and has random agent trainer sampling setup
    """

    mock_builder = MockBuilder(
        global_config=SimpleNamespace(normalize_observations=True), components=[]
    )

    mock_builder.store.unique_net_keys = ["network_0", "network_1", "network_2"]
    mock_builder.store.network_sampling_setup = [
        ["network_0"],
        ["network_1"],
        ["network_2"],
    ]

    return mock_builder


##################
# Mock trainers
##################


class MockTrainer(Trainer):
    """Mock for the trainer"""

    def __init__(
        self,
        store: SimpleNamespace,
        components: List[Callback] = [],
    ) -> None:
        """Creates mock trainer for testing

        Args:
            store : trainer store
            components : list of Mava components
        """

        super().__init__(store, components)
        self.store.agents = ["agent_0", "agent_1", "agent_2"]


@pytest.fixture
def mock_single_trainer_shared_weights_fixed_sampling() -> Trainer:
    """Creates mock trainer

    Has shared weights and fixed agent trainer sampling setup.
    """

    mock_trainer = MockTrainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.trainer_id = "trainer_0"
    mock_trainer.store.table_network_config = {
        "trainer_0": ["network_agent", "network_agent", "network_agent"]
    }

    return mock_trainer


@pytest.fixture
def mock_single_trainer_no_shared_weights_fixed_sampling() -> Trainer:
    """Creates mock trainer

    Doesn't have shared weights and has fixed agent
    trainer sampling setup.
    """

    mock_trainer = MockTrainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.trainer_id = "trainer_0"
    mock_trainer.store.table_network_config = {
        "trainer_0": ["network_agent_0", "network_agent_1", "network_agent_2"]
    }

    return mock_trainer


@pytest.fixture
def mock_single_trainer_no_shared_weights_random_sampling() -> Trainer:
    """Creates mock trainer

    Doesn't have shared weights and has random agent
    trainer sampling setup.
    """

    mock_trainer = MockTrainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.trainer_id = "trainer_0"
    mock_trainer.store.table_network_config = {"trainer_0": ["network_2"]}

    return mock_trainer


@pytest.fixture
def mock_one_trainer_per_network_shared_weights_fixed_sampling() -> Trainer:
    """Creates mock trainer

    Doesn't have shared weights and has fixed agent
    trainer sampling setup.
    """

    mock_trainer = MockTrainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.trainer_id = "trainer_0"
    mock_trainer.store.table_network_config = {
        "trainer_0": ["network_agent", "network_agent", "network_agent"]
    }

    return mock_trainer


@pytest.fixture
def mock_one_trainer_per_network_no_shared_weights_fixed_sampling() -> Trainer:
    """Creates mock trainer

    Doesn't have shared weights and has fixed agent
    trainer sampling setup.
    """

    mock_trainer = MockTrainer(store=SimpleNamespace(), components=[])

    # trainer id must be specified per test
    mock_trainer.store.table_network_config = {
        "trainer_0": ["network_agent_0", "network_agent_1", "network_agent_2"],
        "trainer_1": ["network_agent_0", "network_agent_1", "network_agent_2"],
        "trainer_2": ["network_agent_0", "network_agent_1", "network_agent_2"],
    }

    return mock_trainer


@pytest.fixture
def mock_one_trainer_per_network_random_sampling() -> Trainer:
    """Creates mock trainer

    Doesn't have shared weights and has random agent
    trainer sampling setup.
    """

    mock_trainer = MockTrainer(store=SimpleNamespace(), components=[])

    # trainer id must be given at each test
    mock_trainer.store.table_network_config = {
        "trainer_0": ["network_0"],
        "trainer_1": ["network_1"],
        "trainer_2": ["network_2"],
    }

    return mock_trainer


############################
# TESTS
############################


def check_opt_states(mock_builder: Builder) -> None:
    """Checks optimiser states are as expected

    Args:
        mock_builder : Fixture SystemBuilder
    """
    assert list(mock_builder.store.policy_opt_states.keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert list(mock_builder.store.critic_opt_states.keys()) == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]

    for net_key in mock_builder.store.networks.keys():
        assert (
            mock_builder.store.policy_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                0
            ]
            == optax.EmptyState()
        )
        assert (
            mock_builder.store.critic_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                0
            ]
            == optax.EmptyState()
        )

        assert isinstance(
            mock_builder.store.policy_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                1
            ],
            optax.ScaleByAdamState,
        )
        assert isinstance(
            mock_builder.store.critic_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                1
            ],
            optax.ScaleByAdamState,
        )

        assert mock_builder.store.policy_opt_states[net_key][
            constants.OPT_STATE_DICT_KEY
        ][1][0] == jnp.array([0])
        assert mock_builder.store.critic_opt_states[net_key][
            constants.OPT_STATE_DICT_KEY
        ][1][0] == jnp.array([0])

        assert list(
            mock_builder.store.policy_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                1
            ][1]
        ) == list(
            jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t, dtype=float),
                mock_builder.store.networks[net_key].policy_params,
            )
        )
        assert list(
            mock_builder.store.critic_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                1
            ][1]
        ) == list(
            jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t, dtype=float),
                mock_builder.store.networks[net_key].critic_params,
            )
        )

        assert list(
            mock_builder.store.policy_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                1
            ][2]
        ) == list(
            jax.tree_util.tree_map(
                jnp.zeros_like,
                mock_builder.store.networks[net_key].policy_params,
            )
        )
        assert list(
            mock_builder.store.critic_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                1
            ][2]
        ) == list(
            jax.tree_util.tree_map(
                jnp.zeros_like,
                mock_builder.store.networks[net_key].critic_params,
            )
        )

        assert (
            mock_builder.store.policy_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                2
            ]
            == optax.EmptyState()
        )
        assert (
            mock_builder.store.critic_opt_states[net_key][constants.OPT_STATE_DICT_KEY][
                2
            ]
            == optax.EmptyState()
        )


# ON_BUILDING_INIT_END TESTS


def test_single_trainer_shared_weights_fixed_sampling(
    mock_builder_shared_weights_fixed_sampling: Builder,
    single_trainer_init: SingleTrainerInit,
) -> None:
    """Tests on_building_init_end hook.

    Single trainer for all networks, shared network weights and
    fixed agent network sampling.
    """

    builder = mock_builder_shared_weights_fixed_sampling
    single_trainer_init.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {"network_agent": "agent_0"}
    assert builder.store.table_network_config == {
        "trainer_0": ["network_agent", "network_agent", "network_agent"]
    }
    assert builder.store.trainer_networks == {"trainer_0": ["network_agent"]}
    assert builder.store.networks == builder.store.network_factory()

    check_opt_states(builder)


def test_single_trainer_no_shared_weights_fixed_sampling(
    mock_builder_no_shared_weights_fixed_sampling: Builder,
    single_trainer_init: SingleTrainerInit,
) -> None:
    """Tests on_building_init_end hook.

    Single trainer for all networks, no shared network weights and
    fixed agent network sampling.
    """

    builder = mock_builder_no_shared_weights_fixed_sampling
    single_trainer_init.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {
        "network_agent_0": "agent_0",
        "network_agent_1": "agent_1",
        "network_agent_2": "agent_2",
    }
    assert builder.store.table_network_config == {
        "trainer_0": ["network_agent_0", "network_agent_1", "network_agent_2"]
    }
    assert builder.store.trainer_networks == {
        "trainer_0": ["network_agent_0", "network_agent_1", "network_agent_2"]
    }
    assert builder.store.networks == builder.store.network_factory()

    check_opt_states(builder)


def test_single_trainer_no_shared_weights_random_sampling(
    mock_builder_no_shared_weights_random_sampling: Builder,
    single_trainer_init: SingleTrainerInit,
) -> None:
    """Tests on_building_init_end hook.

    Single trainer for all networks, no shared network weights and
    random agent network sampling.
    """

    builder = mock_builder_no_shared_weights_random_sampling
    single_trainer_init.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {
        "network_0": "agent_0",
        "network_1": "agent_1",
        "network_2": "agent_2",
    }
    assert builder.store.table_network_config == {"trainer_0": ["network_2"]}
    assert builder.store.trainer_networks == {
        "trainer_0": ["network_0", "network_1", "network_2"]
    }
    assert builder.store.networks == builder.store.network_factory()

    check_opt_states(builder)


def test_one_trainer_per_network_shared_weights_fixed_sampling(
    mock_builder_shared_weights_fixed_sampling: Builder,
    one_trainer_per_network_init: OneTrainerPerNetworkInit,
) -> None:
    """Tests on_building_init_end hook.

    One trainer per network, shared network weights and
    fixed agent network sampling.
    """

    builder = mock_builder_shared_weights_fixed_sampling
    one_trainer_per_network_init.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {"network_agent": "agent_0"}
    assert builder.store.table_network_config == {
        "trainer_0": ["network_agent", "network_agent", "network_agent"]
    }
    assert builder.store.trainer_networks == {"trainer_0": ["network_agent"]}
    assert builder.store.networks == builder.store.network_factory()

    check_opt_states(builder)


def test_custom_trainer_on_building_init_value_error(
    mock_builder_no_shared_weights_fixed_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    Custom trainer init. Asserts than exception is thrown when an empty dictionary
    is passed in.
    """

    trainer_init = CustomTrainerInit(config=CustomTrainerInitConfig())

    builder = mock_builder_no_shared_weights_fixed_sampling

    with pytest.raises(ValueError):
        trainer_init.on_building_init_end(builder)

    trainer_init = CustomTrainerInit(
        config=CustomTrainerInitConfig(trainer_networks=None)  # type:ignore
    )

    builder = mock_builder_no_shared_weights_fixed_sampling

    with pytest.raises(ValueError):
        trainer_init.on_building_init_end(builder)


def test_custom_trainer_init_no_shared_weights_random_sampling(
    mock_builder_no_shared_weights_random_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    Custom trainer init. Asserts that custom trainer init works correctly
    when trainer networks corresponding to random network sampling and no
    shared weights are passed in.
    """

    trainer_init = CustomTrainerInit(
        config=CustomTrainerInitConfig(
            trainer_networks={
                "trainer_0": ["network_0"],
                "trainer_1": ["network_1"],
                "trainer_2": ["network_2"],
            }
        )
    )

    builder = mock_builder_no_shared_weights_random_sampling

    trainer_init.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {
        "network_0": "agent_0",
        "network_1": "agent_1",
        "network_2": "agent_2",
    }

    assert builder.store.table_network_config == {
        "trainer_0": ["network_0"],
        "trainer_1": ["network_1"],
        "trainer_2": ["network_2"],
    }

    assert builder.store.networks == builder.store.network_factory()

    check_opt_states(builder)


def test_custom_trainer_init_shared_weights_fixed_sampling(
    mock_builder_shared_weights_fixed_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    Custom trainer init. Asserts that custom trainer init works correctly
    when trainer networks corresponding to fixed network sampling and shared
    network weights are passed in.
    """

    trainer_init = CustomTrainerInit(
        config=CustomTrainerInitConfig(
            trainer_networks={"trainer_0": ["network_agent"]}
        )
    )

    builder = mock_builder_shared_weights_fixed_sampling

    trainer_init.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {"network_agent": "agent_0"}
    assert builder.store.table_network_config == {
        "trainer_0": ["network_agent", "network_agent", "network_agent"]
    }
    assert builder.store.networks == builder.store.network_factory()

    check_opt_states(builder)


#################################
# ON_TRAINING_UTILITY_FNS TESTS
#################################


def test_training_utility_single_trainer_shared_weights(
    mock_single_trainer_shared_weights_fixed_sampling: Trainer,
    single_trainer_init: SingleTrainerInit,
) -> None:
    """Tests on_training_utility_fn hook in TrainerInit

    Tests with single trainer for all agents where network weights are
    shared and the trainer sampling setup is fixed.
    """

    trainer = mock_single_trainer_shared_weights_fixed_sampling

    single_trainer_init.on_training_utility_fns(trainer)

    assert trainer.store.trainer_table_entry == [
        "network_agent",
        "network_agent",
        "network_agent",
    ]
    assert trainer.store.trainer_agents == ["agent_0", "agent_1", "agent_2"]
    assert trainer.store.trainer_agent_net_keys == {
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
    }


def test_training_utility_single_trainer_no_shared_weights(
    mock_single_trainer_no_shared_weights_fixed_sampling: Trainer,
    single_trainer_init: SingleTrainerInit,
) -> None:
    """Tests on_training_utility_fn hook in TrainerInit

    Tests with single trainer for all agents where network weights aren't
    shared and the trainer sampling setup is fixed.
    """

    trainer = mock_single_trainer_no_shared_weights_fixed_sampling

    single_trainer_init.on_training_utility_fns(trainer)

    assert trainer.store.trainer_table_entry == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert trainer.store.trainer_agents == ["agent_0", "agent_1", "agent_2"]
    assert trainer.store.trainer_agent_net_keys == {
        "agent_0": "network_agent_0",
        "agent_1": "network_agent_1",
        "agent_2": "network_agent_2",
    }


def test_training_utility_single_trainer_no_shared_weights_random_sampling(
    mock_single_trainer_no_shared_weights_random_sampling: Trainer,
    single_trainer_init: SingleTrainerInit,
) -> None:
    """Tests on_training_utility_fn hook in TrainerInit

    Tests with single trainer for all agents where network weights aren't
    shared and the trainer sampling setup is random.
    """

    trainer = mock_single_trainer_no_shared_weights_random_sampling

    single_trainer_init.on_training_utility_fns(trainer)

    assert trainer.store.trainer_table_entry == ["network_2"]
    assert trainer.store.trainer_agents == ["agent_0"]
    assert trainer.store.trainer_agent_net_keys == {"agent_0": "network_2"}


def test_training_utility_one_trainer_per_network_shared_weights(
    mock_one_trainer_per_network_shared_weights_fixed_sampling: Trainer,
    one_trainer_per_network_init: OneTrainerPerNetworkInit,
) -> None:
    """Tests on_training_utility_fns hook.

        Tests with one trainer per network where network weights are
    shared and the trainer sampling setup is fixed.
    """

    trainer = mock_one_trainer_per_network_shared_weights_fixed_sampling

    one_trainer_per_network_init.on_training_utility_fns(trainer)

    assert trainer.store.trainer_table_entry == [
        "network_agent",
        "network_agent",
        "network_agent",
    ]
    assert trainer.store.trainer_agents == ["agent_0", "agent_1", "agent_2"]
    assert trainer.store.trainer_agent_net_keys == {
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
    }


def test_training_utility_one_trainer_per_network_no_shared_weights(
    mock_one_trainer_per_network_no_shared_weights_fixed_sampling: Trainer,
    one_trainer_per_network_init: OneTrainerPerNetworkInit,
) -> None:
    """Tests on_training_utility_fns hook.

        Tests with one trainer per network where network weights aren't
    shared and the trainer sampling setup is fixed.
    """

    trainer_0 = mock_one_trainer_per_network_no_shared_weights_fixed_sampling

    trainer_0.store.trainer_id = "trainer_0"

    one_trainer_per_network_init.on_training_utility_fns(trainer_0)

    assert trainer_0.store.trainer_table_entry == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert trainer_0.store.trainer_agents == ["agent_0", "agent_1", "agent_2"]
    assert trainer_0.store.trainer_agent_net_keys == {
        "agent_0": "network_agent_0",
        "agent_1": "network_agent_1",
        "agent_2": "network_agent_2",
    }

    trainer_1 = mock_one_trainer_per_network_no_shared_weights_fixed_sampling

    trainer_1.store.trainer_id = "trainer_1"

    one_trainer_per_network_init.on_training_utility_fns(trainer_1)

    assert trainer_1.store.trainer_table_entry == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert trainer_1.store.trainer_agents == ["agent_0", "agent_1", "agent_2"]
    assert trainer_1.store.trainer_agent_net_keys == {
        "agent_0": "network_agent_0",
        "agent_1": "network_agent_1",
        "agent_2": "network_agent_2",
    }

    trainer_2 = mock_one_trainer_per_network_no_shared_weights_fixed_sampling

    trainer_2.store.trainer_id = "trainer_2"

    one_trainer_per_network_init.on_training_utility_fns(trainer_2)

    assert trainer_2.store.trainer_table_entry == [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    assert trainer_2.store.trainer_agents == ["agent_0", "agent_1", "agent_2"]
    assert trainer_2.store.trainer_agent_net_keys == {
        "agent_0": "network_agent_0",
        "agent_1": "network_agent_1",
        "agent_2": "network_agent_2",
    }


def test_training_utility_one_trainer_per_network_random_sampling(
    mock_one_trainer_per_network_random_sampling: Trainer,
    one_trainer_per_network_init: OneTrainerPerNetworkInit,
) -> None:
    """Tests on_training_utility_fns hook.

        Tests with one trainer per network where network weights aren't
    shared and the trainer sampling setup is random.
    """

    trainer_0 = mock_one_trainer_per_network_random_sampling

    trainer_0.store.trainer_id = "trainer_0"

    one_trainer_per_network_init.on_training_utility_fns(trainer_0)

    assert trainer_0.store.trainer_table_entry == ["network_0"]
    assert trainer_0.store.trainer_agents == ["agent_0"]
    assert trainer_0.store.trainer_agent_net_keys == {"agent_0": "network_0"}

    trainer_1 = mock_one_trainer_per_network_random_sampling

    trainer_1.store.trainer_id = "trainer_1"

    one_trainer_per_network_init.on_training_utility_fns(trainer_1)

    assert trainer_1.store.trainer_table_entry == ["network_1"]
    assert trainer_1.store.trainer_agents == ["agent_0"]
    assert trainer_1.store.trainer_agent_net_keys == {"agent_0": "network_1"}

    trainer_2 = mock_one_trainer_per_network_random_sampling

    trainer_2.store.trainer_id = "trainer_2"

    one_trainer_per_network_init.on_training_utility_fns(trainer_2)

    assert trainer_2.store.trainer_table_entry == ["network_2"]
    assert trainer_2.store.trainer_agents == ["agent_0"]
    assert trainer_2.store.trainer_agent_net_keys == {"agent_0": "network_2"}
