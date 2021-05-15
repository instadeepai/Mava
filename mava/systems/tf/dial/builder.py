import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Type

import reverb
import sonnet as snt
import tensorflow as tf
from acme.tf import variable_utils
from acme.utils import counting, loggers

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.communication import (  # BroadcastedCommunication,
    BaseCommunicationModule,
)
from mava.systems.builders import SystemBuilder
from mava.systems.tf import executors
from mava.systems.tf.dial.execution import DIALExecutor
from mava.systems.tf.dial.training import DIALTrainer


@dataclasses.dataclass
class DIALConfig:
    """Configuration options for the DIAL system
    Args:
        environment_spec: description of the actions, observations, etc.
        networks: the online Q network (the one being optimized)
        batch_size: batch size for updates.
        prefetch_size: size to prefetch from replay.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        min_replay_size: minimum replay size before updating. This and all
            following arguments are related to dataset construction and will be
            ignored if a dataset argument is passed.
        max_replay_size: maximum replay size.
        importance_sampling_exponent: power to which importance weights are raised
            before normalizing.
        priority_exponent: exponent used in prioritized sampling.
        n_step: number of steps to squash into a single transition.
        epsilon: probability of taking a random action; ignored if a policy
            network is given.
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        checkpoint: boolean indicating whether to checkpoint the learner.
        checkpoint_subpath: directory for the checkpoint.
        policy_networks: if given, this will be used as the policy network.
            Otherwise, an epsilon greedy policy using the online Q network will be
            created. Policy network is used in the actor to sample actions.
        max_gradient_norm: used for gradient clipping.
        replay_table_name: string indicating what name to give the replay table.
    """

    environment_spec: specs.MAEnvironmentSpec
    # networks: Dict[str, snt.Module]
    policy_optimizer: snt.Optimizer
    shared_weights: bool = True
    batch_size: int = 1
    prefetch_size: int = 4
    target_update_period: int = 100
    samples_per_insert: float = 32.0
    min_replay_size: int = 1
    max_replay_size: int = 1000000
    importance_sampling_exponent: float = 0.2
    priority_exponent: float = 0.6
    n_step: int = 1
    epsilon: Optional[tf.Tensor] = None
    discount: float = 1.00
    logger: loggers.Logger = None
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    # policy_networks: Optional[Dict[str, snt.Module]] = None
    max_gradient_norm: Optional[float] = None
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    counter: counting.Counter = None
    clipping: bool = False
    # communication_module: BaseCommunicationModule = BroadcastedCommunication
    sequence_length: int = 10
    period: int = 1


class DIALBuilder(SystemBuilder):
    """Builder for DIAL which constructs individual components of the system."""

    """Defines an interface for defining the components of an MARL system.
      Implementations of this interface contain a complete specification of a
      concrete MARL system. An instance of this class can be used to build an
      MARL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(
        self,
        config: DIALConfig,
        executor_fn: Type[core.Executor] = DIALExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """Args:
        config: Configuration options for the DIAL system.
        executor_fn: Executor function to use"""

        self._config = config
        self._extra_specs = extra_specs

        """ _agents: a list of the agent specs (ids).
            _agent_types: a list of the types of agents to be used."""
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()
        self._executor_fn = executor_fn

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""

        # Select adder
        if issubclass(self._executor_fn, executors.FeedForwardExecutor):
            raise ValueError(
                "(dries): Why is there a feedforward executor version of DIAL?"
            )
            adder = reverb_adders.ParallelNStepTransitionAdder.signature(
                environment_spec
            )
        elif issubclass(self._executor_fn, executors.RecurrentExecutor):

            adder = reverb_adders.ParallelEpisodeAdder.signature(
                environment_spec, self._extra_specs
            )
        else:
            print(self._executor_fn)
            raise NotImplementedError("Unknown executor type: ", self._executor_fn)

        return [
            reverb.Table.queue(
                name=self._config.replay_table_name,
                max_size=self._config.max_replay_size,
                signature=adder,
            )
        ]

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the system."""
        dataset = reverb.ReplayDataset.from_table_signature(
            server_address=replay_client.server_address,
            table=self._config.replay_table_name,
            max_in_flight_samples_per_worker=1,
            # sequence_length=self._config.sequence_length,
            emit_timesteps=False,
        )
        dataset = dataset.batch(self._config.batch_size, drop_remainder=True)
        return iter(dataset)

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
          replay_client: Reverb Client which points to the replay server.
        """

        # Select adder
        if issubclass(self._executor_fn, executors.FeedForwardExecutor):
            adder = reverb_adders.ParallelNStepTransitionAdder(
                priority_fns=None,  # {self._config.replay_table_name: lambda x: 1.0},
                client=replay_client,
                n_step=self._config.n_step,
                discount=self._config.discount,
            )
        elif issubclass(self._executor_fn, executors.RecurrentExecutor):
            adder = reverb_adders.ParallelEpisodeAdder(
                priority_fns=None,  # {self._config.replay_table_name: lambda x: 1.0},
                client=replay_client,
                max_sequence_length=self._config.sequence_length,
                # period=self._config.period,
            )
        else:
            raise NotImplementedError("Unknown executor type: ", self._executor_fn)

        return adder

    def make_executor(
        self,
        policy_networks: Dict[str, snt.Module],
        communication_module: BaseCommunicationModule,
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Executor:
        """Create an executor instance.
        Args:
          policy_networks: A struct of instance of all the different policy networks;
           this should be a callable
            which takes as input observations and returns actions.
          adder: How data is recorded (e.g. added to replay).
          variable_source: A source providing the necessary executor parameters.
        """
        shared_weights = self._config.shared_weights

        variable_client = None
        if variable_source:
            agent_keys = self._agent_types if shared_weights else self._agents

            # Create policy variables
            variables = {}
            for agent in agent_keys:
                variables[agent] = policy_networks[agent].variables

            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"policy": variables},
                update_period=10,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the actor which defines how we take actions.
        return self._executor_fn(
            policy_networks=policy_networks,
            communication_module=communication_module,
            shared_weights=shared_weights,
            variable_client=variable_client,
            agent_specs=self._config.environment_spec.get_agent_specs(),
            adder=adder,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        communication_module: BaseCommunicationModule,
        huber_loss_parameter: float = 1.0,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
    ) -> core.Trainer:
        """Creates an instance of the trainer.
        Args:
          networks: struct describing the networks needed by the trainer; this can
            be specific to the trainer in question.
          dataset: iterator over samples from replay.
          replay_client: client which allows communication with replay, e.g. in
            order to update priorities.
          counter: a Counter which allows for recording of counts (trainer steps,
            executor steps, etc.) distributed throughout the system.
          logger: Logger object for logging metadata.
        """
        agents = self._agents
        agent_types = self._agent_types
        shared_weights = self._config.shared_weights
        clipping = self._config.clipping
        discount = self._config.discount
        target_update_period = self._config.target_update_period
        max_gradient_norm = self._config.max_gradient_norm
        importance_sampling_exponent = self._config.importance_sampling_exponent

        # The learner updates the parameters (and initializes them).
        trainer = DIALTrainer(
            agents=agents,
            agent_types=agent_types,
            networks=networks["policies"],
            target_network=networks["target_policies"],
            observation_networks=networks["observations"],
            shared_weights=shared_weights,
            discount=discount,
            importance_sampling_exponent=importance_sampling_exponent,
            policy_optimizer=self._config.policy_optimizer,
            target_update_period=target_update_period,
            dataset=dataset,
            huber_loss_parameter=huber_loss_parameter,
            replay_client=replay_client,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=self._config.checkpoint,
            max_gradient_norm=max_gradient_norm,
            communication_module=communication_module,
        )
        return trainer
