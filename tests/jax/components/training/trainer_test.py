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

from types import SimpleNamespace

import pytest

from mava.components.jax.training.trainer import (
    OneTrainerPerNetworkInit,
    SingleTrainerInit,
)
from mava.systems.jax.builder import Builder
from mava.systems.jax.trainer import Trainer

#################
# Mock builders
#################


@pytest.fixture
def mock_builder_shared_weights_fixed_sampling() -> Builder:
    """Creates mock builder.

    Has shared weights and fixed agent trainer sampling setup.
    """

    mock_builder = Builder(global_config=SimpleNamespace(), components=[])

    mock_builder.store.agents = ["agent_0", "agent_1", "agent_2"]
    mock_builder.store.unique_net_keys = ["network_agent"]
    mock_builder.store.network_sampling_setup = [
        ["network_agent", "network_agent", "network_agent"]
    ]
    mock_builder.store.network_factory = lambda: "network_initialized"

    return mock_builder


@pytest.fixture
def mock_builder_no_shared_weights_fixed_sampling() -> Builder:
    """Create mock builder.

    Doesn't shared weights and has fixed agent trainer sampling setup.
    """

    mock_builder = Builder(global_config=SimpleNamespace(), components=[])

    mock_builder.store.agents = ["agent_0", "agent_1", "agent_2"]
    mock_builder.store.unique_net_keys = [
        "network_agent_0",
        "network_agent_1",
        "network_agent_2",
    ]
    mock_builder.store.network_sampling_setup = [
        ["network_agent_0", "network_agent_1", "network_agent_2"]
    ]
    mock_builder.store.network_factory = lambda: "network_initialized"

    return mock_builder


@pytest.fixture
def mock_builder_no_shared_weights_random_sampling() -> Builder:
    """Create mock builder.

    Doesn't have shared weights and has random agent trainer sampling setup
    """

    mock_builder = Builder(global_config=SimpleNamespace(), components=[])

    mock_builder.store.agents = ["agent_0", "agent_1", "agent_2"]
    mock_builder.store.unique_net_keys = ["network_0", "network_1", "network_2"]
    mock_builder.store.network_sampling_setup = [
        ["network_0"],
        ["network_1"],
        ["network_2"],
    ]
    mock_builder.store.network_factory = lambda: "network_initialized"

    return mock_builder


##################
# Mock trainers
##################


@pytest.fixture
def mock_single_trainer_shared_weights_fixed_sampling() -> Trainer:
    """Creates mock trainer

    Has shared weights and fixed agent trainer sampling setup.
    """

    mock_trainer = Trainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.agents = ["agent_0", "agent_1", "agent_2"]
    mock_trainer.store.trainer_id = "trainer"
    mock_trainer.store.table_network_config = {
        "trainer": ["network_agent", "network_agent", "network_agent"]
    }

    return mock_trainer


@pytest.fixture
def mock_single_trainer_no_shared_weights_fixed_sampling() -> Trainer:
    """Creates mock trainer

    Doesn't have shared weights and has fixed agent
    trainer sampling setup.
    """

    mock_trainer = Trainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.agents = ["agent_0", "agent_1", "agent_2"]
    mock_trainer.store.trainer_id = "trainer"
    mock_trainer.store.table_network_config = {
        "trainer": ["network_agent_0", "network_agent_1", "network_agent_2"]
    }

    return mock_trainer


@pytest.fixture
def mock_single_trainer_no_shared_weights_random_sampling() -> Trainer:
    """Creates mock trainer

    Doesn't have shared weights and has random agent
    trainer sampling setup.
    """

    mock_trainer = Trainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.trainer_id = "trainer"
    mock_trainer.store.agents = ["agent_0", "agent_1", "agent_2"]
    mock_trainer.store.table_network_config = {"trainer": ["network_2"]}

    return mock_trainer


@pytest.fixture
def mock_one_trainer_per_network_shared_weights_fixed_sampling() -> Trainer:
    """Creates mock trainer

    Doesn't have shared weights and has fixed agent
    trainer sampling setup.
    """

    mock_trainer = Trainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.agents = ["agent_0", "agent_1", "agent_2"]
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

    mock_trainer = Trainer(store=SimpleNamespace(), components=[])

    mock_trainer.store.agents = ["agent_0", "agent_1", "agent_2"]
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

    mock_trainer = Trainer(store=SimpleNamespace(), components=[])

    # trainer id must be given at each test
    mock_trainer.store.agents = ["agent_0", "agent_1", "agent_2"]
    mock_trainer.store.table_network_config = {
        "trainer_0": ["network_0"],
        "trainer_1": ["network_1"],
        "trainer_2": ["network_2"],
    }

    return mock_trainer


############################
# TESTS
############################

# ON_BUILDING_INIT_END TESTS


def test_single_trainer_shared_weights_fixed_sampling(
    mock_builder_shared_weights_fixed_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    Single trainer for all networks, shared network weights and
    fixed agent network sampling.
    """

    mock_trainer = SingleTrainerInit()

    builder = mock_builder_shared_weights_fixed_sampling
    mock_trainer.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {"network_agent": "agent_0"}
    assert builder.store.table_network_config == {
        "trainer": ["network_agent", "network_agent", "network_agent"]
    }
    assert builder.store.trainer_networks == {"trainer": ["network_agent"]}
    assert builder.store.networks == "network_initialized"


def test_single_trainer_no_shared_weights_fixed_sampling(
    mock_builder_no_shared_weights_fixed_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    Single trainer for all networks, no shared network weights and
    fixed agent network sampling.
    """

    mock_trainer = SingleTrainerInit()

    builder = mock_builder_no_shared_weights_fixed_sampling
    mock_trainer.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {
        "network_agent_0": "agent_0",
        "network_agent_1": "agent_1",
        "network_agent_2": "agent_2",
    }
    assert builder.store.table_network_config == {
        "trainer": ["network_agent_0", "network_agent_1", "network_agent_2"]
    }
    assert builder.store.trainer_networks == {
        "trainer": ["network_agent_0", "network_agent_1", "network_agent_2"]
    }
    assert builder.store.networks == "network_initialized"


def test_single_trainer_no_shared_weights_random_sampling(
    mock_builder_no_shared_weights_random_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    Single trainer for all networks, no shared network weights and
    random agent network sampling.
    """

    mock_trainer = SingleTrainerInit()

    builder = mock_builder_no_shared_weights_random_sampling
    mock_trainer.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {
        "network_0": "agent_0",
        "network_1": "agent_1",
        "network_2": "agent_2",
    }
    assert builder.store.table_network_config == {"trainer": ["network_2"]}
    assert builder.store.trainer_networks == {
        "trainer": ["network_0", "network_1", "network_2"]
    }
    assert builder.store.networks == "network_initialized"


def test_one_trainer_per_network_shared_weights_fixed_sampling(
    mock_builder_shared_weights_fixed_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    One trainer per network, shared network weights and
    fixed agent network sampling.
    """

    mock_trainer = OneTrainerPerNetworkInit()

    builder = mock_builder_shared_weights_fixed_sampling
    mock_trainer.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {"network_agent": "agent_0"}
    assert builder.store.table_network_config == {
        "trainer_0": ["network_agent", "network_agent", "network_agent"]
    }
    assert builder.store.trainer_networks == {"trainer_0": ["network_agent"]}
    assert builder.store.networks == "network_initialized"


def test_one_trainer_per_network_no_shared_weights_fixed_sampling(
    mock_builder_no_shared_weights_fixed_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    One trainer per network, no shared network weights and
    fixed agent network sampling.
    """

    mock_trainer = OneTrainerPerNetworkInit()

    builder = mock_builder_no_shared_weights_fixed_sampling
    mock_trainer.on_building_init_end(builder)

    assert builder.store.net_spec_keys == {
        "network_agent_0": "agent_0",
        "network_agent_1": "agent_1",
        "network_agent_2": "agent_2",
    }
    assert builder.store.table_network_config == {
        "trainer_0": ["network_agent_0", "network_agent_1", "network_agent_2"],
        "trainer_1": ["network_agent_0", "network_agent_1", "network_agent_2"],
        "trainer_2": ["network_agent_0", "network_agent_1", "network_agent_2"],
    }
    assert builder.store.trainer_networks == {
        "trainer_0": ["network_agent_0"],
        "trainer_1": ["network_agent_1"],
        "trainer_2": ["network_agent_2"],
    }
    assert builder.store.networks == "network_initialized"


def test_one_trainer_per_network_random_sampling(
    mock_builder_no_shared_weights_random_sampling: Builder,
) -> None:
    """Tests on_building_init_end hook.

    One trainer per network, no shared network weights and
    random agent network sampling.
    """

    mock_trainer = OneTrainerPerNetworkInit()

    builder = mock_builder_no_shared_weights_random_sampling
    mock_trainer.on_building_init_end(builder)

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
    assert builder.store.trainer_networks == {
        "trainer_0": ["network_0"],
        "trainer_1": ["network_1"],
        "trainer_2": ["network_2"],
    }
    assert builder.store.networks == "network_initialized"


#################################
# ON_TRAINING_UTILITY_FNS TESTS
#################################


def test_training_utility_single_trainer_shared_weights(
    mock_single_trainer_shared_weights_fixed_sampling: Trainer,
) -> None:
    """Tests on_training_utility_fn hook in TrainerInit

    Tests with single trainer for all agents where network weights are
    shared and the trainer sampling setup is fixed.
    """

    trainer = mock_single_trainer_shared_weights_fixed_sampling

    trainer_init = SingleTrainerInit()

    trainer_init.on_training_utility_fns(trainer)

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
) -> None:
    """Tests on_training_utility_fn hook in TrainerInit

    Tests with single trainer for all agents where network weights aren't
    shared and the trainer sampling setup is fixed.
    """

    trainer = mock_single_trainer_no_shared_weights_fixed_sampling

    trainer_init = SingleTrainerInit()

    trainer_init.on_training_utility_fns(trainer)

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
) -> None:
    """Tests on_training_utility_fn hook in TrainerInit

    Tests with single trainer for all agents where network weights aren't
    shared and the trainer sampling setup is random.
    """

    trainer = mock_single_trainer_no_shared_weights_random_sampling

    trainer_init = SingleTrainerInit()

    trainer_init.on_training_utility_fns(trainer)

    assert trainer.store.trainer_table_entry == ["network_2"]
    assert trainer.store.trainer_agents == ["agent_0"]
    assert trainer.store.trainer_agent_net_keys == {"agent_0": "network_2"}


def test_training_utility_one_trainer_per_network_shared_weights(
    mock_one_trainer_per_network_shared_weights_fixed_sampling: Trainer,
) -> None:
    """Tests on_training_utility_fns hook.

        Tests with one trainer per network where network weights are
    shared and the trainer sampling setup is fixed.
    """

    trainer = mock_one_trainer_per_network_shared_weights_fixed_sampling

    trainer_init = OneTrainerPerNetworkInit()

    trainer_init.on_training_utility_fns(trainer)

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
) -> None:
    """Tests on_training_utility_fns hook.

        Tests with one trainer per network where network weights aren't
    shared and the trainer sampling setup is fixed.
    """

    trainer_init = OneTrainerPerNetworkInit()

    trainer_0 = mock_one_trainer_per_network_no_shared_weights_fixed_sampling

    trainer_0.store.trainer_id = "trainer_0"

    trainer_init.on_training_utility_fns(trainer_0)

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

    trainer_init.on_training_utility_fns(trainer_1)

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

    trainer_init.on_training_utility_fns(trainer_2)

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
) -> None:
    """Tests on_training_utility_fns hook.

        Tests with one trainer per network where network weights aren't
    shared and the trainer sampling setup is random.
    """
    trainer_init = OneTrainerPerNetworkInit()

    trainer_0 = mock_one_trainer_per_network_random_sampling

    trainer_0.store.trainer_id = "trainer_0"

    trainer_init.on_training_utility_fns(trainer_0)

    assert trainer_0.store.trainer_table_entry == ["network_0"]
    assert trainer_0.store.trainer_agents == ["agent_0"]
    assert trainer_0.store.trainer_agent_net_keys == {"agent_0": "network_0"}

    trainer_1 = mock_one_trainer_per_network_random_sampling

    trainer_1.store.trainer_id = "trainer_1"

    trainer_init.on_training_utility_fns(trainer_1)

    assert trainer_1.store.trainer_table_entry == ["network_1"]
    assert trainer_1.store.trainer_agents == ["agent_0"]
    assert trainer_1.store.trainer_agent_net_keys == {"agent_0": "network_1"}

    trainer_2 = mock_one_trainer_per_network_random_sampling

    trainer_2.store.trainer_id = "trainer_2"

    trainer_init.on_training_utility_fns(trainer_2)

    assert trainer_2.store.trainer_table_entry == ["network_2"]
    assert trainer_2.store.trainer_agents == ["agent_0"]
    assert trainer_2.store.trainer_agent_net_keys == {"agent_0": "network_2"}
