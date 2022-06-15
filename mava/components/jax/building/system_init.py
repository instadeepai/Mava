from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from mava.components.jax import Component
from mava.core_jax import SystemBuilder
from mava.utils import enums
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num


@dataclass
class FixedNetworkSystemInitConfig:
    shared_weights: bool = True
    network_sampling_setup: enums.NetworkSampler = (
        enums.NetworkSampler.fixed_agent_networks
    )


class FixedNetworkSystemInit(Component):
    def __init__(
        self, config: FixedNetworkSystemInitConfig = FixedNetworkSystemInitConfig()
    ):
        """Agents will always be assigned to a fixed network.

        A single network get assigned to all agents of the same type
        if weights are shared, else a seperate network is assigned to each agent

        Args:
            config : a dataclass specifying the component parameters
        """
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Compute and add network sampling information to the builder."""

        if (
            self.config.network_sampling_setup
            != enums.NetworkSampler.fixed_agent_networks
        ):
            raise ValueError(
                "Fixed network system init requires fixed_agent_networks sampling"
            )

        builder.store.agent_net_keys = {
            agent: f"network_{agent.split('_')[0]}"
            if self.config.shared_weights
            else f"network_{agent}"
            for agent in builder.store.agents
        }
        builder.store.network_sampling_setup = [
            [
                builder.store.agent_net_keys[key]
                for key in sort_str_num(builder.store.agent_net_keys.keys())
            ]
        ]

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(builder.store.network_sampling_setup[0])
        agent_ids = builder.store.environment_spec.get_agent_ids()
        assert len(agent_ids) == len(builder.store.agent_net_keys.keys())

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(builder.store.agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(builder.store.network_sampling_setup)):
            assert len(builder.store.network_sampling_setup[i]) == sample_length

        # Get all the unique agent network keys
        all_samples = []
        for sample in builder.store.network_sampling_setup:
            all_samples.extend(sample)
        builder.store.unique_net_keys = list(sort_str_num(list(set(all_samples))))

        # Create mapping from ints to networks
        builder.store.net_keys_to_ids = {
            net_key: i for i, net_key in enumerate(builder.store.unique_net_keys)
        }

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "system_init"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return FixedNetworkSystemInitConfig


@dataclass
class RandomSamplingSystemInitConfig:
    network_sampling_setup: enums.NetworkSampler = (
        enums.NetworkSampler.random_agent_networks
    )
    shared_weights: bool = False


class RandomSamplingSystemInit(Component):
    def __init__(
        self, config: RandomSamplingSystemInitConfig = RandomSamplingSystemInitConfig()
    ):
        """Create N network policies, where N is the number of agents.

        Randomly select policies from this set for each agent at the start of a
        episode. This sampling is done with replacement so the same policy
        can be selected for more than one agent for a given episode.

        Args:
            config : a dataclass specifying the component parameters.
        """
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Compute and add network sampling information to the builder."""

        if (
            self.config.network_sampling_setup
            != enums.NetworkSampler.random_agent_networks
        ):
            raise ValueError(
                "Random sampling system init requires random_agent_networks sampling"
            )

        if builder.store.shared_weights:
            raise ValueError(
                "Shared weights cannot be used with random policy per agent"
            )
        builder.store.agent_net_keys = {
            builder.store.agents[i]: f"network_{i}"
            for i in range(len(builder.store.agents))
        }

        builder.store.network_sampling_setup = [
            [
                [builder.store.agent_net_keys[key]]
                for key in sort_str_num(builder.store.agent_net_keys.keys())
            ]
        ]

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(builder.store.network_sampling_setup[0])
        agent_ids = builder.store.environment_spec.get_agent_ids()
        assert len(agent_ids) == len(builder.store.agent_net_keys.keys())

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(builder.store.agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(builder.store.network_sampling_setup)):
            assert len(builder.store.network_sampling_setup[i]) == sample_length

        # Get all the unique agent network keys
        all_samples = []
        for sample in builder.store.network_sampling_setup:
            all_samples.extend(sample)
        builder.store.unique_net_keys = list(sort_str_num(list(set(all_samples))))

        # Create mapping from ints to networks
        builder.store.net_keys_to_ids = {
            net_key: i for i, net_key in enumerate(builder.store.unique_net_keys)
        }

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "system_init"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return RandomSamplingSystemInitConfig


@dataclass
class CustomSamplingSystemInitConfig:
    network_sampling_setup: Union[List, enums.NetworkSampler] = field(
        default_factory=lambda: []
    )
    shared_weights: bool = False


class CustomSamplingSystemInit(Component):
    def __init__(
        self,
        config: CustomSamplingSystemInitConfig = CustomSamplingSystemInitConfig(),
    ):
        """Specify a custom network sampling setup.

        This network sampling setup will then get used to randomly select
        policies for agents.

        Args:
            config : a dataclass specifying the component parameters.
        """
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Compute and add network sampling information to the builder."""

        if self.config.network_sampling_setup == []:
            raise ValueError("A custom network sampling setup list must be provided.")

        # Use network_sampling_setup to determine agent_net_keys
        _, builder.store.agent_net_keys = sample_new_agent_keys(
            builder.store.agents,
            self.config.network_sampling_setup,
        )

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(self.config.network_sampling_setup[0])
        agent_ids = builder.store.environment_spec.get_agent_ids()
        assert len(agent_ids) == len(builder.store.agent_net_keys.keys())

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(builder.store.agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(self.config.network_sampling_setup)):
            assert len(self.config.network_sampling_setup[i]) == sample_length

        # Get all the unique agent network keys
        all_samples = []
        for sample in self.config.network_sampling_setup:
            all_samples.extend(sample)
        builder.store.unique_net_keys = list(sort_str_num(list(set(all_samples))))

        # Create mapping from ints to networks
        builder.store.net_keys_to_ids = {
            net_key: i for i, net_key in enumerate(builder.store.unique_net_keys)
        }

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "system_init"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return CustomSamplingSystemInitConfig
