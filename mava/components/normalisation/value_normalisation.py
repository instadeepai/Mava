from dataclasses import dataclass

from mava import constants
from mava.core_jax import SystemBuilder, SystemParameterServer, SystemTrainer
from mava.utils.jax_training_utils import (
    compute_running_mean_var_count,
    init_norm_params,
)

from .base_normalisation import BaseNormalisation


@dataclass
class ValueNormalisationConfig:
    normalise_target_values: bool = True


class ValueNormalisation(BaseNormalisation):
    def __init__(
        self, config: ValueNormalisationConfig = ValueNormalisationConfig()
    ) -> None:
        """Initialising ValueNormalisation"""
        super().__init__(config)

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Create the norm_params dict for holding normalisation parameters"""
        builder.store.norm_params = {}

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Initialise target value normalisation parameters"""
        if self.config.normalise_target_values:
            values_norm_key = constants.VALUES_NORM_STATE_DICT_KEY
            builder.store.norm_params[values_norm_key] = {}
            for agent in builder.store.agents:
                builder.store.norm_params[values_norm_key][agent] = init_norm_params((1,))

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Initialises observation normalisation parameters

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """
        if self.config.normalise_target_values:
            trainer.store.target_running_stats_fn = compute_running_mean_var_count

    def on_parameter_server_init(self, server: SystemParameterServer) -> None:
        """Stores the normalisation parameters in the parameter server"""
        server.store.parameters["norm_params"] = server.store.norm_params

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "value_normalisation"
