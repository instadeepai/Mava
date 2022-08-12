# Creating your own component
If a desired functionality does not exist in Mava, it can easily be extended by creating a new component. In order to create component a class must be created that inherits from the base [`Component`](https://github.com/instadeepai/Mava/blob/7b11a082ba790e1b2c2f0acd633ff605fffbe768/mava/components/jax/component.py#L24) class with the relevant hooks overwritten. TODO (docs) EXPLAIN THE STORE. As an example, consider the following component which creates a function for computing the generalized advantage estimate:

```python
@dataclass
class GAEConfig:
    gae_lambda: float = 0.95
    max_abs_reward: float = np.inf


class GAE(Utility):
    def __init__(
        self,
        config: GAEConfig = GAEConfig(),
    ):
        """Component defines advantage estimation function.

        Args:
            config: GAEConfig.
        """
        self.config = config

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Create and store a GAE advantage function.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """

        def gae_advantages(
            rewards: jnp.ndarray, discounts: jnp.ndarray, values: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Use truncated GAE to compute advantages.

            Args:
                rewards: Agent rewards.
                discounts: Agent discount factors.
                values: Agent value estimations.

            Returns:
                Tuple of advantage values, target values.
            """

            # Apply reward clipping.
            max_abs_reward = self.config.max_abs_reward
            rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

            advantages = rlax.truncated_generalized_advantage_estimation(
                rewards[:-1], discounts[:-1], self.config.gae_lambda, values
            )
            advantages = jax.lax.stop_gradient(advantages)

            # Exclude the bootstrap value
            target_values = values[:-1] + advantages
            target_values = jax.lax.stop_gradient(target_values)

            return advantages, target_values

        trainer.store.gae_fn = gae_advantages

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "gae_fn"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return GAEConfig
```
