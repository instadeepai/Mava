from types import SimpleNamespace

from mava import constants
from mava.components.training.trainer import BaseTrainerInit
from mava.core_jax import SystemBuilder
from mava.utils.sort_utils import sort_str_num


class TrainerInit(BaseTrainerInit):
    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Set up the networks during the build."""
        unique_net_keys = builder.store.unique_net_keys

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in builder.store.trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        builder.store.net_spec_keys = {}
        for i in range(len(unique_net_keys)):
            builder.store.net_spec_keys[unique_net_keys[i]] = builder.store.agents[
                i % len(builder.store.agents)
            ]

        # Setup table_network_config
        builder.store.table_network_config = {}
        for trainer_key in builder.store.trainer_networks.keys():
            most_matches = 0
            trainer_nets = builder.store.trainer_networks[trainer_key]
            for sample in builder.store.network_sampling_setup:
                matches = 0
                for entry in sample:
                    if entry in trainer_nets:
                        matches += 1
                if most_matches < matches:
                    matches = most_matches
                    builder.store.table_network_config[trainer_key] = sample

        # TODO (Matthew): networks need to be created on the nodes instead?
        builder.store.networks = builder.store.network_factory()

        # Wrap opt_states in a mutable type (dict) since optax return an immutable tuple
        builder.store.policy_opt_states = {}
        for net_key in builder.store.networks.keys():
            builder.store.policy_opt_states[net_key] = {
                constants.OPT_STATE_DICT_KEY: builder.store.policy_optimiser.init(
                    builder.store.networks[net_key].policy_params
                )
            }  # pytype: disable=attribute-error


class SingleTrainerInit(TrainerInit):
    def __init__(self, config: SimpleNamespace = SimpleNamespace()):
        """Initialises a single trainer.

        Single trainer is used to train all networks.

        Args:
            config : a dataclass specifying the component parameters.
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Assigns trainers to networks for training.

        Args:
            builder : the system builder
        Raises:
            ValueError: Raises an error when trainer_networks is not
                        set to single_trainer.
        """
        # Setup trainer_networks
        unique_net_keys = builder.store.unique_net_keys
        builder.store.trainer_networks = {"trainer_0": unique_net_keys}
        super(SingleTrainerInit, self).on_building_init_end(builder)
