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

"""MADDPG scaled system builder implementation."""

import copy
import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import reverb
import sonnet as snt
import tensorflow as tf
from acme import datasets
from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers
from dm_env import specs as dm_specs

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.systems.tf import executors, variable_utils
from mava.systems.tf.maddpg import training
from mava.systems.tf.maddpg.execution import MADDPGFeedForwardExecutor
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.utils.sort_utils import sort_str_num
from mava.wrappers import NetworkStatisticsActorCritic, ScaledDetailedTrainerStatistics

BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray


@dataclasses.dataclass
class MADDPGConfig:
    """Configuration options for the MADDPG system.
    Args:
        environment_spec: description of the action and observation spaces etc. for
            each agent in the system.
        policy_optimizer: optimizer(s) for updating policy networks.
        critic_optimizer: optimizer for updating critic networks.
        agent_net_keys: (dict, optional): specifies what network each agent uses.
            Defaults to {}.
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
        sigma: Gaussian sigma parameter.
        checkpoint: boolean to indicate whether to checkpoint models.
        checkpoint_subpath: subdirectory specifying where to store checkpoints.
        replay_table_name: string indicating what name to give the replay table."""

    environment_spec: specs.MAEnvironmentSpec
    policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]]
    critic_optimizer: snt.Optimizer
    num_executors: int
    agent_net_keys: Dict[str, str]
    trainer_networks: Dict[str, List]
    table_network_config: Dict[str, List]
    executor_samples: List
    net_to_ints: Dict[str, int]
    unique_net_keys: List[str]
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
    bootstrap_n: int = 10
    max_gradient_norm: Optional[float] = None
    sigma: float = 0.3
    logger: loggers.Logger = None
    counter: counting.Counter = None
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    replay_table_name: str = "trainer"  # reverb_adders.DEFAULT_PRIORITY_TABLE


class MADDPGBuilder:
    """Builder for scaled MADDPG which constructs individual components of the
    system."""

    def __init__(
        self,
        config: MADDPGConfig,
        trainer_fn: Union[
            Type[training.MADDPGBaseTrainer],
            Type[training.MADDPGBaseRecurrentTrainer],
        ] = training.MADDPGDecentralisedTrainer,
        executor_fn: Type[core.Executor] = MADDPGFeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """Initialise the system.
        Args:
            config (MADDPGConfig): system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn (Union[ Type[training.MADDPGBaseTrainer],
                Type[training.MADDPGBaseRecurrentTrainer], ], optional): Trainer
                function, of a correpsonding type to work with the selected system
                architecture. Defaults to training.MADDPGDecentralisedTrainer.
            executor_fn (Type[core.Executor], optional): Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to MADDPGFeedForwardExecutor.
            extra_specs (Dict[str, Any], optional): defines the specifications of extra
                information used by the system. Defaults to {}.
        """

        self._config = config
        self._extra_specs = extra_specs
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()
        self._trainer_fn = trainer_fn
        self._executor_fn = executor_fn

    def convert_discrete_to_bounded(
        self, environment_spec: specs.MAEnvironmentSpec
    ) -> specs.MAEnvironmentSpec:
        """convert discrete action space to bounded continuous action space
        Args:
            environment_spec (specs.MAEnvironmentSpec): description of
                the action, observation spaces etc. for each agent in the system.
        Returns:
            specs.MAEnvironmentSpec: updated environment spec.
        """

        env_adder_spec: specs.MAEnvironmentSpec = copy.deepcopy(environment_spec)
        keys = env_adder_spec._keys
        for key in keys:
            agent_spec = env_adder_spec._specs[key]
            if type(agent_spec.actions) == DiscreteArray:
                num_actions = agent_spec.actions.num_values
                minimum = [float("-inf")] * num_actions
                maximum = [float("inf")] * num_actions
                new_act_spec = BoundedArray(
                    shape=(num_actions,),
                    minimum=minimum,
                    maximum=maximum,
                    dtype="float32",
                    name="actions",
                )

                env_adder_spec._specs[key] = EnvironmentSpec(
                    observations=agent_spec.observations,
                    actions=new_act_spec,
                    rewards=agent_spec.rewards,
                    discounts=agent_spec.discounts,
                )
        return env_adder_spec

    def covert_specs(self, spec: Dict[str, Any], num_networks: int) -> Dict[str, Any]:
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
        """ "Create tables to insert data into.
        Args:
            environment_spec (specs.MAEnvironmentSpec): description of the action and
                observation spaces etc. for each agent in the system.
        Raises:
            NotImplementedError: unknown executor type.
        Returns:
            List[reverb.Table]: a list of data tables for inserting data.
        """

        # Select adder
        env_adder_spec = self.convert_discrete_to_bounded(environment_spec)

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
        for t_i in range(len(self._config.table_network_config.keys())):
            # TODO (dries): Clean the below coverter code up.
            # Convert a Mava spec
            num_networks = len(self._config.table_network_config[f"trainer_{t_i}"])
            env_spec = copy.deepcopy(env_adder_spec)
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
                    name=f"{self._config.replay_table_name}_{t_i}",
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
            replay_client (reverb.Client): Reverb Client which points to the
                replay server.
        Returns:
            [type]: dataset iterator.
        Yields:
            Iterator[reverb.ReplaySample]: data samples from the dataset.
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
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
            replay_client (reverb.Client): Reverb Client which points to the
                replay server.
        Raises:
            NotImplementedError: unknown executor type.
        Returns:
            Optional[adders.ParallelAdder]: adder which sends data to a replay buffer.
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
                int_to_nets=self._config.unique_net_keys,
                n_step=self._config.n_step,
                table_network_config=self._config.table_network_config,
                discount=self._config.discount,
            )
        elif issubclass(self._executor_fn, executors.RecurrentExecutor):
            adder = reverb_adders.ParallelSequenceAdder(
                priority_fns=priority_fns,
                client=replay_client,
                int_to_nets=self._config.unique_net_keys,
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
            variables (Dict[str, snt.Variable]): dictionary with variable_source
            variables in.
        Returns:
            variables (Dict[str, snt.Variable]): dictionary with variable_source
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
            networks (Dict[str, Dict[str, snt.Module]]): dictionary with the
            system's networks in.
        Returns:
            variable_source (MavaVariableSource): A Mava variable source object.
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
            variables, self._config.checkpoint, self._config.checkpoint_subpath
        )
        return variable_source

    def make_executor(
        self,
        # executor_id: str,
        networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[MavaVariableSource] = None,
    ) -> core.Executor:
        """Create an executor instance.
        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            adder (Optional[adders.ParallelAdder], optional): adder to send data to
                a replay buffer. Defaults to None.
            variable_source (Optional[core.VariableSource], optional): variables server.
                Defaults to None.
        Returns:
            core.Executor: system executor, a collection of agents making up the part
                of the system generating data by interacting the environment.
        """
        # Create policy variables
        variables = {}
        get_keys = []
        for net_type_key in ["observations", "policies"]:
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

        variable_client = None
        if variable_source:
            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables=variables,
                get_keys=get_keys,
                get_period=self._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.get_and_wait()

        # Create the actor which defines how we take actions.
        return self._executor_fn(
            policy_networks=policy_networks,
            counts=counts,
            net_to_ints=self._config.net_to_ints,
            agent_specs=self._config.environment_spec.get_agent_specs(),
            agent_net_keys=self._config.agent_net_keys,
            executor_samples=self._config.executor_samples,
            variable_client=variable_client,
            adder=adder,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        variable_source: MavaVariableSource,
        trainer_networks: List[Any],
        trainer_table_entry: List[Any],
        logger: Optional[types.NestedLogger] = None,
        connection_spec: Dict[str, List[str]] = None,
    ) -> core.Trainer:
        """Create a trainer instance.
        Args:
            networks (Dict[str, Dict[str, snt.Module]]): system networks.
            dataset (Iterator[reverb.ReplaySample]): dataset iterator to feed data to
                the trainer networks.
            counter (Optional[counting.Counter], optional): a Counter which allows for
                recording of counts, e.g. trainer steps. Defaults to None.
            logger (Optional[types.NestedLogger], optional): Logger object for logging
                metadata. Defaults to None.
            connection_spec (Dict[str, List[str]], optional): connection topology used
                for networked system architectures. Defaults to None.
        Returns:
            core.Trainer: system trainer, that uses the collected data from the
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
        for net_type_key in networks.keys():
            for net_key in networks[net_type_key].keys():
                variables[f"{net_key}_{net_type_key}"] = networks[net_type_key][
                    net_key
                ].variables
                if net_key in set(trainer_networks):
                    set_keys.append(f"{net_key}_{net_type_key}")
                else:
                    get_keys.append(f"{net_key}_{net_type_key}")

        variables = self.create_counter_variables(variables)
        num_steps = variables["trainer_steps"]

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
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "observation_networks": networks["observations"],
            "target_policy_networks": networks["target_policies"],
            "target_critic_networks": networks["target_critics"],
            "target_observation_networks": networks["target_observations"],
            "agent_net_keys": trainer_agent_net_keys,
            "policy_optimizer": self._config.policy_optimizer,
            "critic_optimizer": self._config.critic_optimizer,
            "max_gradient_norm": max_gradient_norm,
            "discount": discount,
            "target_averaging": target_averaging,
            "target_update_period": target_update_period,
            "target_update_rate": target_update_rate,
            "variable_client": variable_client,
            "dataset": dataset,
            "counts": counts,
            "num_steps": num_steps,
            "logger": logger,
        }
        if connection_spec:
            trainer_config["connection_spec"] = connection_spec

        if issubclass(self._trainer_fn, training.MADDPGBaseRecurrentTrainer):
            trainer_config["bootstrap_n"] = self._config.bootstrap_n

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(**trainer_config)

        # NB If using both NetworkStatistics and TrainerStatistics, order is important.
        # NetworkStatistics needs to appear before TrainerStatistics.
        # TODO(Kale-ab/Arnu): need to fix wrapper type issues
        trainer = NetworkStatisticsActorCritic(trainer)  # type: ignore

        trainer = ScaledDetailedTrainerStatistics(  # type: ignore
            trainer, metrics=["policy_loss", "critic_loss"]
        )

        return trainer
