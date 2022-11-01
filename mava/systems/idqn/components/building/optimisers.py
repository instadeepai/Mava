from dataclasses import dataclass
from typing import Optional

import optax
from optax._src import base as optax_base
from mava.components.building.optimisers import Optimisers

from mava.core_jax import SystemBuilder

@dataclass
class OptimisersConfig:
    policy_learning_rate: float = 1e-3
    adam_epsilon: float = 1e-5
    max_gradient_norm: float = 0.5
    policy_optimiser: Optional[optax_base.GradientTransformation] = None


class Optimiser(Optimisers):
    def __init__(
        self,
        config: OptimisersConfig = OptimisersConfig(),
    ):
        """Component defines the default way to initialise optimisers.

        Args:
            config: DefaultOptimisers.
        """
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Create and store the optimisers.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        # Build the optimiser function here
        if not self.config.policy_optimiser:
            builder.store.policy_optimiser = optax.chain(
                optax.clip_by_global_norm(self.config.max_gradient_norm),
                optax.scale_by_adam(eps=self.config.adam_epsilon),
                optax.scale(-self.config.policy_learning_rate),
            )
        else:
            builder.store.policy_optimiser = self.config.policy_optimiser