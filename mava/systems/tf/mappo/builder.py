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

"""MAPPO system builder implementation."""

import copy
import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import reverb
import sonnet as snt
import tensorflow as tf

# from acme.adders.reverb.sequence import EndBehavior
from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.systems.tf import variable_utils
from mava.systems.tf.mappo import execution, training
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.utils.sort_utils import sort_str_num
from mava.wrappers import NetworkStatisticsActorCritic, ScaledDetailedTrainerStatistics


@dataclasses.dataclass
class MAPPOConfig:
    """Configuration options for the MAPPO system

    Args:
        environment_spec: description of the action and observation spaces etc. for
            each agent in the system.
        policy_optimizer: optimizer(s) for updating policy networks.
        critic_optimizer: optimizer for updating critic networks. This is not
            used if using single optim.
        agent_net_keys: (dict, optional): specifies what network each agent uses.
            Defaults to {}.
        trainer_networks: networks each trainer trains on.
        table_network_config: Networks each table (trainer) expects.
        network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
        fix_sampler: Optional list that can fix the executor sampler to sample
                in a specific way.
        net_spec_keys: Optional network to agent mapping used to get the environment
                specs for each network.
        net_keys_to_ids: mapping from net_key to network id.
        unique_net_keys: list of unique net_keys.
        checkpoint_minute_interval (int): The number of minutes to wait between
            checkpoints.
        sequence_length: recurrent sequence rollout length.
        sequence_period: consecutive starting points for overlapping rollouts across a
            sequence. Defaults to sequence length -1.
        discount: discount to use for TD updates.
        lambda_gae: scalar determining the mix of bootstrapping vs further accumulation
            of multi-step returns at each timestep. See `High-Dimensional Continuous
            Control Using Generalized Advantage Estimation` for more information.
        max_queue_size: maximum number of items in the queue. Should be
            larger than batch size.
        executor_variable_update_period: the rate at which executors sync their
            paramters with the trainer.
        batch_size: batch size for updates.
        entropy_cost: contribution of entropy regularization to the total loss.
        baseline_cost: contribution of the value loss to the total loss.
        clipping_epsilon: Hyper-parameter for clipping in the policy objective. Roughly:
            how far can the new policy go from the old policy while still profiting?
            The new policy can still go farther than the clip_ratio says, but it doesn't
            help on the objective anymore.
        max_gradient_norm: value to specify the maximum clipping value for the gradient
            norm during optimization.
        checkpoint: boolean to indicate whether to checkpoint models.
        checkpoint_subpath: subdirectory specifying where to store checkpoints.
        termination_condition: An optional terminal condition can be provided
            that stops the program once the condition is satisfied. Available options
            include specifying maximum values for trainer_steps, trainer_walltime,
            evaluator_steps, evaluator_episodes, executor_episodes or executor_steps.
            E.g. termination_condition = {'trainer_steps': 100000}.
        replay_table_name: string indicating what name to give the replay table.
        evaluator_interval: intervals that evaluator are run at.
        learning_rate_scheduler_fn: function/class that takes in a trainer step t
                and returns the current learning rate.
    """

    environment_spec: specs.EnvironmentSpec
    policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]]
    critic_optimizer: snt.Optimizer
    agent_net_keys: Dict[str, str]
    trainer_networks: Dict[str, List]
    table_network_config: Dict[str, List]
    network_sampling_setup: List
    fix_sampler: Optional[List]
    net_spec_keys: Dict[str, str]
    net_keys_to_ids: Dict[str, int]
    unique_net_keys: List[str]
    checkpoint_minute_interval: int
    sequence_length: int = 10
    sequence_period: int = 9
    discount: float = 0.99
    lambda_gae: float = 0.95
    max_queue_size: Optional[int] = 1000
    executor_variable_update_period: int = 100
    batch_size: int = 512
    minibatch_size: Optional[int] = None
    num_epochs: int = 10
    entropy_cost: float = 0.01
    baseline_cost: float = 1.0
    clipping_epsilon: float = 0.2
    max_gradient_norm: Optional[float] = None
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    termination_condition: Optional[Dict[str, int]] = None
    learning_rate_scheduler_fn: Optional[Any] = None
    evaluator_interval: Optional[dict] = None
    normalize_advantage: bool = False


class MAPPOBuilder:
    """Builder for MAPPO which constructs individual components of the system."""

    def __init__(
        self,
        config: MAPPOConfig,
        trainer_fn: Type[training.MAPPOTrainer] = training.MAPPOTrainer,
        executor_fn: Type[core.Executor] = execution.MAPPOFeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
    ):
        """Initialise the system.

        Args:
            config (MAPPOConfig): system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn (Type[training.MAPPOTrainer], optional): Trainer
                function, of a correpsonding type to work with the selected system
                architecture. Defaults to training.MAPPOTrainer.
            executor_fn (Type[core.Executor], optional): Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to execution.MAPPOFeedForwardExecutor.
        """

        self._config = config
        self._agents = self._config.environment_spec.get_agent_ids()
        self._trainer_fn = trainer_fn
        self._executor_fn = executor_fn
        self._extra_specs = extra_specs

    def add_log_prob_to_spec(
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
            new_act_spec = {"actions": agent_spec.actions}

            # Make dummy logits
            new_act_spec["log_probs"] = tf.ones(shape=(), dtype=tf.float32)

            env_adder_spec._specs[key] = EnvironmentSpec(
                observations=agent_spec.observations,
                actions=new_act_spec,
                rewards=agent_spec.rewards,
                discounts=agent_spec.discounts,
            )
        return env_adder_spec

    def get_nets_specific_specs(
        self, spec: Dict[str, Any], trainer_network_names: List
    ) -> Dict[str, Any]:
        """Convert specs.
        Args:
            spec: agent specs
            trainer_network_names: names of the networks in the desired reverb table
            (to get specs for)
        Returns:
            distilled version of agent specs containing only specs related to
            `networks_names`
        """
        if type(spec) is not dict:
            return spec

        agents = []
        for network in trainer_network_names:
            agents.append(self._config.net_spec_keys[network])

        agents = sort_str_num(agents)
        converted_spec: Dict[str, Any] = {}
        if agents[0] in spec.keys():
            for i, agent in enumerate(agents):
                converted_spec[self._agents[i]] = spec[agent]
        else:
            # For the extras
            for key in spec.keys():
                converted_spec[key] = self.get_nets_specific_specs(
                    spec[key], trainer_network_names
                )
        return converted_spec

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into.

        Args:
            environment_spec (specs.MAEnvironmentSpec): description of the action and
                observation spaces etc. for each agent in the system.

        Returns:
            List[reverb.Table]: a list of data tables for inserting data.
        """

        # Create system architecture with target networks.
        adder_env_spec = self.add_log_prob_to_spec(environment_spec)

        replay_tables = []
        for table_key in self._config.table_network_config.keys():
            # TODO (dries): Clean the below converter code up.
            # Convert a Mava spec
            trainer_network_names = self._config.table_network_config[table_key]
            env_spec = copy.deepcopy(adder_env_spec)
            env_spec._specs = self.get_nets_specific_specs(
                env_spec._specs, trainer_network_names
            )

            env_spec._keys = list(sort_str_num(env_spec._specs.keys()))
            if env_spec.extra_specs is not None:
                env_spec.extra_specs = self.get_nets_specific_specs(
                    env_spec.extra_specs, trainer_network_names
                )
            extra_specs = self.get_nets_specific_specs(
                self._extra_specs,
                trainer_network_names,
            )

            signature = reverb_adders.ParallelSequenceAdder.signature(
                env_spec,
                sequence_length=self._config.sequence_length,
                extras_spec=extra_specs,
            )

            replay_tables.append(
                reverb.Table.queue(
                    name=table_key,
                    max_size=self._config.max_queue_size,
                    signature=signature,
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

        # NOTE: From https://github.com/deepmind/acme/blob/6bf350df1d9dd16cd85217908ec9f47553278976/acme/agents/jax/ppo/builder.py#L89  # noqa: E501
        # We don't use datasets.make_reverb_dataset() here to avoid interleaving
        # and prefetching, that doesn't work well with can_sample() check on update.

        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=replay_client.server_address,
            table=table_name,
            max_in_flight_samples_per_worker=2 * self._config.batch_size,
        )
        # Add batch dimension.
        dataset = dataset.batch(self._config.batch_size, drop_remainder=True)
        return dataset.as_numpy_iterator()

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ReverbParallelAdder]:
        """Create an adder which records data generated by the executor/environment.

        Args:
            replay_client (reverb.Client): Reverb Client which points to the
                replay server.

        Raises:
            NotImplementedError: unknown executor type.

        Returns:
            Optional[adders.ReverbParallelAdder]: adder which sends data to a
            replay buffer.
        """

        # Create custom priority functons for the adder
        priority_fns = {
            table_key: lambda x: 1.0
            for table_key in self._config.table_network_config.keys()
        }

        return reverb_adders.ParallelSequenceAdder(
            priority_fns=priority_fns,
            client=replay_client,
            net_ids_to_keys=self._config.unique_net_keys,
            table_network_config=self._config.table_network_config,
            period=self._config.sequence_period,
            sequence_length=self._config.sequence_length,
            # end_of_episode_behavior=EndBehavior.CONTINUE,
        )
        # Note (dries): Using end_of_episode_behavior=EndBehavior.CONTINUE can
        # break the adders when using multiple trainers.

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
        # executor_id: str,
        networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        adder: Optional[adders.ReverbParallelAdder] = None,
        variable_source: Optional[MavaVariableSource] = None,
        evaluator: bool = False,
    ) -> core.Executor:
        """Create an executor instance.
        Args:
            policy_networks: policy networks for each agent in
                the system.
            adder : adder to send data to
                a replay buffer. Defaults to None.
            variable_source: variables server.
                Defaults to None.
            evaluator: boolean indicator if the executor is used for
                for evaluation only.

        Returns:
            system executor, a collection of agents making up the part
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
        evaluator_interval = self._config.evaluator_interval if evaluator else None
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

        # Create the actor which defines how we take actions.
        return self._executor_fn(
            policy_networks=policy_networks,
            counts=counts,
            net_keys_to_ids=self._config.net_keys_to_ids,
            agent_specs=self._config.environment_spec.get_agent_specs(),
            agent_net_keys=self._config.agent_net_keys,
            network_sampling_setup=self._config.network_sampling_setup,
            fix_sampler=self._config.fix_sampler,
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
        connection_spec: Dict[str, List[str]] = None,
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
            connection_spec: connection topology used
                for networked system architectures. Defaults to None.
        Returns:
            system trainer, that uses the collected data from the
                executors to update the parameters of the agent networks in the system.
        """
        # This assumes agents are sort_str_num in the other methods
        max_gradient_norm = self._config.max_gradient_norm

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
            update_period=1,
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
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "observation_networks": networks["observations"],
            "agent_net_keys": trainer_agent_net_keys,
            "critic_optimizer": self._config.critic_optimizer,
            "policy_optimizer": self._config.policy_optimizer,
            "max_gradient_norm": max_gradient_norm,
            "discount": self._config.discount,
            "minibatch_size": self._config.minibatch_size,
            "num_epochs": self._config.num_epochs,
            "variable_client": variable_client,
            "dataset": dataset,
            "counts": counts,
            "logger": logger,
            "lambda_gae": self._config.lambda_gae,
            "normalize_advantage": self._config.normalize_advantage,
            "entropy_cost": self._config.entropy_cost,
            "baseline_cost": self._config.baseline_cost,
            "clipping_epsilon": self._config.clipping_epsilon,
            "learning_rate_scheduler_fn": self._config.learning_rate_scheduler_fn,
        }

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(**trainer_config)

        # NB If using both NetworkStatistics and TrainerStatistics, order is important.
        # NetworkStatistics needs to appear before TrainerStatistics.
        # TODO(Kale-ab/Arnu): need to fix wrapper type issues
        trainer = NetworkStatisticsActorCritic(trainer)  # type: ignore

        trainer = ScaledDetailedTrainerStatistics(  # type: ignore
            trainer, metrics=["policy_loss", "critic_loss", "total_loss"]
        )

        return trainer
