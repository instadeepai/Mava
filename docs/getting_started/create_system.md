# Creating a system in Mava

> 🚧 **Note:** This only applies to the callback redesign of Mava.

In order to create a new system in Mava, a system class must be defined that inherits from the base [`System`][system] class. The `design` method must then be overwritten to return a [`DesignSpec`][design_spec] object containing all the components to be used by a particular system. A default system config may also be created as a `dataclass` which contains the default system hyperparameters to be used.

Please consider the following example where an IPPO system is created:

```python
class IPPOSystem(System):
    def design(self) -> Tuple[DesignSpec, Any]:
        """Design for IPPO system.
        Returns:
            system: system callback components
            default_params: system default parameters
        """
        # Set the default configs
        default_params = IPPODefaultConfig()

        # Default system processes
        # System initialization
        system_init = DesignSpec(
            environment_spec=building.EnvironmentSpec,
            system_init=building.FixedNetworkSystemInit,
        ).get()

        # Executor
        executor_process = DesignSpec(
            executor_init=executing.ExecutorInit,
            executor_observe=executing.FeedforwardExecutorObserve,
            executor_select_action=executing.FeedforwardExecutorSelectAction,
            executor_adder=building.ParallelSequenceAdder,
            adder_priority=building.UniformAdderPriority,
            executor_environment_loop=building.ParallelExecutorEnvironmentLoop,
            networks=building.DefaultNetworks,
            optimisers=building.DefaultOptimisers,
        ).get()

        # Trainer
        trainer_process = DesignSpec(
            trainer_init=training.SingleTrainerInit,
            gae_fn=training.GAE,
            loss=training.MAPGWithTrustRegionClippingLoss,
            epoch_update=training.MAPGEpochUpdate,
            minibatch_update=training.MAPGMinibatchUpdate,
            sgd_step=training.MAPGWithTrustRegionStep,
            step=training.DefaultTrainerStep,
            trainer_dataset=building.TrajectoryDataset,
        ).get()

        # Data Server
        data_server_process = DesignSpec(
            data_server=building.OnPolicyDataServer,
            data_server_adder_signature=building.ParallelSequenceAdderSignature,
            extras_spec=ExtrasLogProbSpec,
        ).get()

        # Parameter Server
        parameter_server_process = DesignSpec(
            parameter_server=updating.DefaultParameterServer,
            executor_parameter_client=building.ExecutorParameterClient,
            trainer_parameter_client=building.TrainerParameterClient,
            termination_condition=updating.CountConditionTerminator,
            checkpointer=updating.Checkpointer,
        ).get()

        system = DesignSpec(
            **system_init,
            **data_server_process,
            **parameter_server_process,
            **executor_process,
            **trainer_process,
            distributor=building.Distributor,
            logger=building.Logger,
            component_dependency_guardrails=ComponentDependencyGuardrails,
        )
        return system, default_params
```

In the above example certain processes are grouped together, which has been done for readability but it is not strictly required. For an example of how a full system may be launched on a particular environment with logging included, please see [here](https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo.py).

When building the system, `system.build` can contain arguments that will overwrite the default config values from any existing component in the system. Commonly overwritten build arguments are:

- `environment_factory` - used to construct the environment (defined in the EnvironmentSpec Component [here](https://github.com/instadeepai/Mava/blob/develop/mava/components/jax/building/environments.py#L39).

- `network_factory` - used to construct the agent networks.(defined in the EnvironmentSpec Component [here](https://github.com/instadeepai/Mava/blob/develop/mava/components/jax/building/networks.py#L34).

- `logger_factory` - used to construct the loggers.

- `experiment_path` - the folder for saving experiment results and restoring a checkpoint from a previous experiment. **To load a pre-trained checkpoint, this variable must point to an existing folder that contains the checkpoint folder generated from a previous experiment.**

- `policy_optimiser` - the optimiser used by the trainer to updated the policy weights.

- `critic_optimiser` - the optimiser used by the trainer to updated the critic weights.

- `run_evaluator` - a flag indicating whether a separate environment process should be run that tracks the system's performance using Tensorboard and possibly create agent gameplay recordings.

- `sample_batch_size` - the batch size to use in the trainer when updating the agent networks.

- `num_epochs` - the number of epochs to train on sampled data before discarding it.

- `num_executors` - the number of experience generators (workers) to run in parallel.

- `multi_process` - determines whether the code is run using multiple distributed processes or using a single process. The multiple processor setup is used for faster training in particular.

- `record_every` - determines how often the evaluator should record an agent gameplay video.

[system]: https://github.com/instadeepai/Mava/blob/7b11a082ba790e1b2c2f0acd633ff605fffbe768/mava/systems/jax/system.py#L28
[design_spec]: https://github.com/instadeepai/Mava/blob/7b11a082ba790e1b2c2f0acd633ff605fffbe768/mava/specs.py#L161
