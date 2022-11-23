# Creating your own component



If a desired functionality does not exist in Mava, a system can easily be extended by creating a new component. In order to create a component, a class must be created that inherits from the base [`Component`][component] class with the relevant hooks overwritten.

Each component requires:

* An associated `dataclass` which defines a component specific config with parameters to be used by it. Parameters from this component config `dataclass` can also be overwritten at a system level.
* A `name` static method which defines the component name. This name gets used to verify that two components with the same functionaltiy are not simultaneously present in the system.
* Defining any relevant hooks to be overwritten by that component where each overwritten hook takes the process associated with a particular component as its input.

The component can be added to the system:

* By adding it to the system design directly (see [here](https://github.com/instadeepai/Mava/blob/develop/mava/systems/ippo/system.py))
* Via system.add() if it is an entirely new component
* Via system.update() if it overrides an existing component with the same name (see [here](https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_with_monitoring.py#L92)).

As an example, please consider the following component which creates a function for computing the generalised advantage estimate and adds that function to the trainer store so that it may be executed later. Notice that this component inherits from `Component` via the class called [`Utility`](https://github.com/instadeepai/Mava/blob/7b11a082ba790e1b2c2f0acd633ff605fffbe768/mava/components/jax/training/base.py#L50).

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
            rewards: jnp.ndarray, discount: jnp.ndarray, values: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Use truncated GAE to compute advantages.

            Args:
                rewards: Agent rewards.
                discount: Agent took a valid step in the environment.
                values: Agent value estimations.

            Returns:
                Tuple of advantage values, target values.
            """

            # Apply reward clipping.
            max_abs_reward = self.config.max_abs_reward
            rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

            advantages = rlax.truncated_generalized_advantage_estimation(
                rewards[:-1], discount[:-1], self.config.gae_lambda, values
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
```

[component]: https://github.com/instadeepai/Mava/blob/7b11a082ba790e1b2c2f0acd633ff605fffbe768/mava/components/jax/component.py#L24
