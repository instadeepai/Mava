# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MADQN scaled system builder implementation."""

import copy
import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import reverb
import sonnet as snt
import tensorflow as tf
from acme import datasets
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers
from dm_env import specs as dm_specs

from mava import Trainer, adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationScheduler,
    BaseExplorationTimestepScheduler,
    ConstantScheduler,
)
from mava.systems.tf import executors, variable_utils
from mava.systems.tf.madqn import training
from mava.systems.tf.madqn.execution import MADQNFeedForwardExecutor
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.utils.builder_utils import initialize_epsilon_schedulers
from mava.utils.sort_utils import sort_str_num
from mava.wrappers import ScaledDetailedTrainerStatistics

BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray


@dataclasses.dataclass
class MADQNConfig:
    """Configuration options for the MADQN system.

    Args:
        environment_spec: description of the action and observation spaces etc. for
            each agent in the system.
        optimizer: optimizer(s) for updating value networks.
        num_executors: number of parallel executors to use.
        agent_net_keys: specifies what network each agent uses.
        trainer_networks: networks each trainer trains on.
        table_network_config: Networks each table (trainer) expects.
        network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
        net_keys_to_ids: mapping from net_key to network id.
        unique_net_keys: list of unique net_keys.
        checkpoint_minute_interval: The number of minutes to wait between
            checkpoints.
        discount: discount to use for TD updates.
        batch_size: batch size for updates.
        prefetch_size: size to prefetch from replay.
        target_averaging: whether to use polyak averaging for target network updates.
        target_update_period: number of steps before target networks are updated.
        target_update_rate: update rate when using averaging.
        executor_variable_update_period: the rate at which executors sync their
            paramters with the trainer.
        min_replay_size: minimum replay size before updating.
        max_replay_size: maximum replay size.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        n_step: number of steps to include prior to boostrapping.
        sequence_length: recurrent sequence rollout length.
        period: consecutive starting points for overlapping rollouts across a sequence.
        max_gradient_norm: value to specify the maximum clipping value for the gradient
            norm during optimization.
        logger: logger to use.
        checkpoint: boolean to indicate whether to checkpoint models.
        checkpoint_subpath: subdirectory specifying where to store checkpoints.
        termination_condition: An optional terminal condition can be provided
            that stops the program once the condition is satisfied. Available options
            include specifying maximum values for trainer_steps, trainer_walltime,
            evaluator_steps, evaluator_episodes, executor_episodes or executor_steps.
            E.g. termination_condition = {'trainer_steps': 100000}.
        learning_rate_scheduler_fn: dict with two functions/classes (one for the
                policy and one for the critic optimizer), that takes in a trainer
                step t and returns the current learning rate,
                e.g. {"policy": policy_lr_schedule ,"critic": critic_lr_schedule}.
                See
                examples/debugging/simple_spread/feedforward/decentralised/run_maddpg_lr_schedule.py
                for an example.
        evaluator_interval: An optional condition that is used to
            evaluate/test system performance after [evaluator_interval]
            condition has been met.
    """

    environment_spec: specs.MAEnvironmentSpec
    optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]]
    num_executors: int
    agent_net_keys: Dict[str, str]
    trainer_networks: Dict[str, List]
    table_network_config: Dict[str, List]
    network_sampling_setup: List
    net_keys_to_ids: Dict[str, int]
    unique_net_keys: List[str]
    checkpoint_minute_interval: int
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_averaging: bool = False
    target_update_period: int = 100
    target_update_rate: Optional[float] = None
    executor_variable_update_period: int = 1000
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: Optional[float] = 32.0
    n_step: int = 5
    sequence_length: int = 20
    period: int = 20
    # bootstrap_n: int = 10
    max_gradient_norm: Optional[float] = None
    logger: loggers.Logger = None
    counter: counting.Counter = None
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    termination_condition: Optional[Dict[str, int]] = None
    evaluator_interval: Optional[dict] = None
    learning_rate_scheduler_fn: Optional[Any] = None


class MADQNBuilder:
    """Builder for MADQN."""

    def __init__(
        self,
        config: MADQNConfig,
        trainer_fn: Type[Trainer] = training.MADQNTrainer,
        executor_fn: Type[core.Executor] = MADQNFeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """Initialise the builder.

        Args:
            config: system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn: Trainer function, of a correpsonding type to work with
                the selected system architecture.
            executor_fn: Executor function, of a corresponding type to work with
                the selected system architecture.
            extra_specs: defines the specifications of extra
                information used by the system.
        """

        self._config = config
        self._extra_specs = extra_specs
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()
        self._trainer_fn = trainer_fn
        self._executor_fn = executor_fn

    def covert_specs(self, spec: Dict[str, Any], num_networks: int) -> Dict[str, Any]:
        """Convert specs.

        Args:
            spec: agent specs
            num_networks: the number of networks

        Returns:
            converted specs
        """
        if type(spec) is not dict:
            return spec

        agents = sort_str_num(self._config.agent_net_keys.keys())[:num_networks]
        converted_spec: Dict[str, Any] = {}
        if agents[0] in spec.keys():
            for agent in agents:
                converted_spec[agent] = spec[agent]
        else:
            # For the extras
            for key in spec.keys():
                converted_spec[key] = self.covert_specs(spec[key], num_networks)
        return converted_spec

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into.

        Args:
            environment_spec: description of the action and
                observation spaces etc. for each agent in the system.

        Raises:
            NotImplementedError: unknown executor type.

        Returns:
            a list of data tables for inserting data.
        """

        if issubclass(self._executor_fn, executors.FeedForwardExecutor):

            def adder_sig_fn(
                env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
            ) -> Any:
                return reverb_adders.ParallelNStepTransitionAdder.signature(
                    env_spec, extra_specs
                )

        elif issubclass(self._executor_fn, executors.RecurrentExecutor):

            def adder_sig_fn(
                env_spec: specs.MAEnvironmentSpec, extra_specs: Dict[str, Any]
            ) -> Any:
                return reverb_adders.ParallelSequenceAdder.signature(
                    env_spec, self._config.sequence_length, extra_specs
                )

        else:
            raise NotImplementedError("Unknown executor type: ", self._executor_fn)

        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            def limiter_fn() -> reverb.rate_limiters:
                return reverb.rate_limiters.MinSize(self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance

            def limiter_fn() -> reverb.rate_limiters:
                return reverb.rate_limiters.SampleToInsertRatio(
                    min_size_to_sample=self._config.min_replay_size,
                    samples_per_insert=self._config.samples_per_insert,
                    error_buffer=error_buffer,
                )

        # Create table per trainer
        replay_tables = []
        for table_key in self._config.table_network_config.keys():
            # TODO (dries): Clean the below coverter code up.
            # Convert a Mava spec
            num_networks = len(self._config.table_network_config[table_key])
            env_spec = copy.deepcopy(environment_spec)
            env_spec._specs = self.covert_specs(env_spec._specs, num_networks)

            env_spec._keys = list(sort_str_num(env_spec._specs.keys()))
            if env_spec.extra_specs is not None:
                env_spec.extra_specs = self.covert_specs(
                    env_spec.extra_specs, num_networks
                )
            extra_specs = self.covert_specs(
                self._extra_specs,
                num_networks,
            )

            replay_tables.append(
                reverb.Table(
                    name=table_key,
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=self._config.max_replay_size,
                    rate_limiter=limiter_fn(),
                    signature=adder_sig_fn(env_spec, extra_specs),
                )
            )
        return replay_tables

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
        table_name: str,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for training/updating the system.

        Args:
            replay_client: Reverb Client which points to the
                replay server.

        Returns:
            [type]: dataset iterator.

        Yields:
            data samples from the dataset.
        """

        sequence_length = (
            self._config.sequence_length
            if issubclass(self._executor_fn, executors.RecurrentExecutor)
            else None
        )

        """Create a dataset iterator to use for learning/updating the system."""
        dataset = datasets.make_reverb_dataset(
            table=table_name,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
            sequence_length=sequence_length,
        )
        return iter(dataset)

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ReverbParallelAdder]:
        """Create an adder which records data generated by the executor/environment.

        Args:
            replay_client: Reverb Client which points to the
                replay server.

        Raises:
            NotImplementedError: unknown executor type.

        Returns:
            adder which sends data to a replay buffer.
        """
        # Create custom priority functons for the adder
        priority_fns = {
            table_key: lambda x: 1.0
            for table_key in self._config.table_network_config.keys()
        }

        # Select adder
        if issubclass(self._executor_fn, executors.FeedForwardExecutor):
            adder = reverb_adders.ParallelNStepTransitionAdder(
                priority_fns=priority_fns,
                client=replay_client,
                net_ids_to_keys=self._config.unique_net_keys,
                n_step=self._config.n_step,
                table_network_config=self._config.table_network_config,
                discount=self._config.discount,
            )
        elif issubclass(self._executor_fn, executors.RecurrentExecutor):
            adder = reverb_adders.ParallelSequenceAdder(
                priority_fns=priority_fns,
                client=replay_client,
                net_ids_to_keys=self._config.unique_net_keys,
                sequence_length=self._config.sequence_length,
                table_network_config=self._config.table_network_config,
                period=self._config.period,
            )
        else:
            raise NotImplementedError("Unknown executor type: ", self._executor_fn)

        return adder

    def create_counter_variables(
        self, variables: Dict[str, tf.Variable]
    ) -> Dict[str, tf.Variable]:
        """Create counter variables.

        Args:
            variables: dictionary with variable_source
            variables in.

        Returns:
            variables: dictionary with variable_source
            variables in.
        """
        variables["trainer_steps"] = tf.Variable(0, dtype=tf.int32)
        variables["trainer_walltime"] = tf.Variable(0, dtype=tf.float32)
        variables["evaluator_steps"] = tf.Variable(0, dtype=tf.int32)
        variables["evaluator_episodes"] = tf.Variable(0, dtype=tf.int32)
        variables["executor_episodes"] = tf.Variable(0, dtype=tf.int32)
        variables["executor_steps"] = tf.Variable(0, dtype=tf.int32)
        return variables

    def make_variable_server(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
    ) -> MavaVariableSource:
        """Create the variable server.

        Args:
            networks: dictionary with the
            system's networks in.

        Returns:
            variable_source: A Mava variable source object.
        """
        # Create variables
        variables = {}
        # Network variables
        for net_type_key in networks.keys():
            for net_key in networks[net_type_key].keys():
                # Ensure obs and target networks are sonnet modules
                variables[f"{net_key}_{net_type_key}"] = tf2_utils.to_sonnet_module(
                    networks[net_type_key][net_key]
                ).variables

        variables = self.create_counter_variables(variables)

        # Create variable source
        variable_source = MavaVariableSource(
            variables,
            self._config.checkpoint,
            self._config.checkpoint_subpath,
            self._config.checkpoint_minute_interval,
            self._config.termination_condition,
        )
        return variable_source

    def make_executor(
        self,
        networks: Dict[str, snt.Module],
        exploration_schedules: Dict[
            str,
            Union[
                BaseExplorationTimestepScheduler,
                BaseExplorationScheduler,
                ConstantScheduler,
            ],
        ],
        adder: Optional[adders.ReverbParallelAdder] = None,
        variable_source: Optional[MavaVariableSource] = None,
        evaluator: bool = False,
        seed: Optional[int] = None,
    ) -> core.Executor:
        """Create an executor instance.

        Args:
            networks: dictionary with the system's networks in.
            policy_networks: policy networks for each agent in
                the system.
            adder: adder to send data to
                a replay buffer. Defaults to None.
            variable_source: variables server.
                Defaults to None.
            evaluator: boolean indicator if the executor is used for
                for evaluation only.
            seed: seed for reproducible sampling.

        Returns:
            system executor, a collection of agents making up the part
                of the system generating data by interacting the environment.
        """
        # Create variables
        variables = {}
        get_keys = []
        for net_type_key in ["observations", "values"]:
            for net_key in networks[net_type_key].keys():
                var_key = f"{net_key}_{net_type_key}"
                variables[var_key] = networks[net_type_key][net_key].variables
                get_keys.append(var_key)
        variables = self.create_counter_variables(variables)

        count_names = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
            "executor_episodes",
            "executor_steps",
        ]
        get_keys.extend(count_names)
        counts = {name: variables[name] for name in count_names}

        evaluator_interval = self._config.evaluator_interval if evaluator else None
        variable_client = None
        if variable_source:
            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables=variables,
                get_keys=get_keys,
                # If we are using evaluator_intervals,
                # we should always get the latest variables.
                update_period=0
                if evaluator_interval
                else self._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.get_and_wait()

        # Pass scheduler and initialize action selectors
        action_selectors_with_scheduler = initialize_epsilon_schedulers(
            exploration_schedules,
            networks["selectors"],
            self._config.agent_net_keys,
            seed=seed,
        )

        # Create the actor which defines how we take actions.
        return self._executor_fn(
            observation_networks=networks["observations"],
            value_networks=networks["values"],
            action_selectors=action_selectors_with_scheduler,
            counts=counts,
            net_keys_to_ids=self._config.net_keys_to_ids,
            agent_specs=self._config.environment_spec.get_agent_specs(),
            agent_net_keys=self._config.agent_net_keys,
            network_sampling_setup=self._config.network_sampling_setup,
            variable_client=variable_client,
            adder=adder,
            evaluator=evaluator,
            interval=evaluator_interval,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        variable_source: MavaVariableSource,
        trainer_networks: List[Any],
        trainer_table_entry: List[Any],
        logger: Optional[types.NestedLogger] = None,
    ) -> core.Trainer:
        """Create a trainer instance.

        Args:
            networks: system networks.
            dataset: dataset iterator to feed data to
                the trainer networks.
            variable_source: Source with variables in.
            trainer_networks: Set of unique network keys to train on..
            trainer_table_entry: List of networks per agent to train on.
            logger: Logger object for logging  metadata.

        Returns:
            system trainer, that uses the collected data from the
                executors to update the parameters of the agent networks in the system.
        """
        # This assumes agents are sort_str_num in the other methods
        agent_types = self._agent_types
        max_gradient_norm = self._config.max_gradient_norm
        discount = self._config.discount
        target_update_period = self._config.target_update_period
        target_averaging = self._config.target_averaging
        target_update_rate = self._config.target_update_rate

        # Create variable client
        variables = {}
        set_keys = []
        get_keys = []
        # TODO (dries): Only add the networks this trainer is working with.
        # Not all of them.
        for net_type_key in ["observations", "values"]:
            for net_key in networks[net_type_key].keys():
                variables[f"{net_key}_{net_type_key}"] = networks[net_type_key][
                    net_key
                ].variables
                if net_key in set(trainer_networks):
                    set_keys.append(f"{net_key}_{net_type_key}")
                else:
                    get_keys.append(f"{net_key}_{net_type_key}")

        variables = self.create_counter_variables(variables)

        count_names = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
            "executor_episodes",
            "executor_steps",
        ]
        get_keys.extend(count_names)
        counts = {name: variables[name] for name in count_names}

        variable_client = variable_utils.VariableClient(
            client=variable_source,
            variables=variables,
            get_keys=get_keys,
            set_keys=set_keys,
            update_period=10,
        )

        # Get all the initial variables
        variable_client.get_all_and_wait()

        # Convert network keys for the trainer.
        trainer_agents = self._agents[: len(trainer_table_entry)]
        trainer_agent_net_keys = {
            agent: trainer_table_entry[a_i] for a_i, agent in enumerate(trainer_agents)
        }
        trainer_config: Dict[str, Any] = {
            "agents": trainer_agents,
            "agent_types": agent_types,
            "value_networks": networks["values"],
            "observation_networks": networks["observations"],
            "target_value_networks": networks["target_values"],
            "target_observation_networks": networks["target_observations"],
            "agent_net_keys": trainer_agent_net_keys,
            "optimizer": self._config.optimizer,
            "max_gradient_norm": max_gradient_norm,
            "discount": discount,
            "target_averaging": target_averaging,
            "target_update_period": target_update_period,
            "target_update_rate": target_update_rate,
            "variable_client": variable_client,
            "dataset": dataset,
            "counts": counts,
            "logger": logger,
            "learning_rate_scheduler_fn": self._config.learning_rate_scheduler_fn,
        }

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(**trainer_config)  # type: ignore

        trainer = ScaledDetailedTrainerStatistics(  # type: ignore
            trainer, metrics=["value_loss"]
        )

        return trainer
