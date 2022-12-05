from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

from mava import constants
from mava.core_jax import SystemBuilder, SystemParameterServer, SystemTrainer
from mava.utils.jax_training_utils import (
    construct_norm_axes_list,
    init_norm_params,
    update_and_normalize_observations,
)

from .base_normalisation import BaseNormalisation


@dataclass
class ObservationNormalisationConfig:
    normalise_observations: bool = True
    normalize_axes: Optional[List[float]] = field(default_factory=lambda: None)


class ObservationNormalisation(BaseNormalisation):
    def __init__(
        self, config: ObservationNormalisationConfig = ObservationNormalisationConfig()
    ) -> None:
        """Initialising ObservationNormalisation"""
        super().__init__(config)

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Create the norm_params dict for holding normalisation parameters"""
        builder.store.norm_params = {}

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Initialise observations' normalisation parameters"""
        obs_norm_key = constants.OBS_NORM_STATE_DICT_KEY
        agent_env_specs = builder.store.ma_environment_spec._agent_environment_specs
        builder.store.norm_params[obs_norm_key] = {}

        for agent in builder.store.agents:
            obs_shape = agent_env_specs[agent].observations.observation.shape

            if self.config.normalise_observations and len(obs_shape) > 1:
                raise NotImplementedError(
                    "Observations normalization only works for 1D feature spaces!"
                )

            builder.store.norm_params[obs_norm_key][agent] = init_norm_params(obs_shape)

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Initialises observation normalisation function"""
        if self.config.normalise_observations:
            observation_stats = trainer.store.norm_params[
                constants.OBS_NORM_STATE_DICT_KEY
            ]

            obs_shape = list(observation_stats.values())[0]["mean"].shape
            norm_axes = construct_norm_axes_list(
                trainer.store.obs_normalisation_start,
                self.config.normalize_axes,
                obs_shape,
            )
            trainer.store.norm_obs_running_stats_fn = partial(
                update_and_normalize_observations,
                axes=norm_axes,
            )

    def on_parameter_server_init(self, server: SystemParameterServer) -> None:
        """Stores the normalisation parameters in the parameter server"""
        server.store.parameters["norm_params"] = server.store.norm_params

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "observation_normalisation"
