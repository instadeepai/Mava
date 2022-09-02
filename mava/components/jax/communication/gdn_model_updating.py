from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Optional, Type

import optax
from optax._src import base as optax_base

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.core_jax import SystemTrainer


class TrainingStateGdn(NamedTuple):
    """Training state consists of network parameters and optimiser state."""

    params: Any
    opt_state: optax.OptState


@dataclass
class GdnTrainerConfig:
    gdn_learning_rate: float = 1e-3
    gdn_adam_epsilon: float = 1e-5
    gdn_max_gradient_norm: float = 0.5
    gdn_optimizer: Optional[optax_base.GradientTransformation] = (None,)


class GdnTrainer(Component):
    def __init__(
        self,
        config: GdnTrainerConfig = GdnTrainerConfig(),
    ):
        """Handles creation of optimisers for GDN.

        Args:
            config: GdnTrainerConfig.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Create optimizer and opt states for GDN.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """
        # Create optimizer
        if not self.config.gdn_optimizer:
            trainer.store.gdn_optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gdn_adam_epsilon),
                optax.scale_by_adam(eps=self.config.gdn_max_gradient_norm),
                optax.scale(-self.config.gdn_learning_rate),
            )
        else:
            trainer.store.gdn_optimizer = self.config.gdn_optimizer

        # Initialize optimizer
        trainer.store.gdn_opt_state = trainer.store.gdn_optimizer.init(
            trainer.store.gdn_network.params
        )

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return GdnTrainerConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        TODO: complete this.

        Returns:
            List of required component classes.
        """
        return []

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "gdn_trainer"
