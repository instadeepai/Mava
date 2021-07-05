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

"""MADDPG system builder implementation."""

import copy
import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import reverb
import sonnet as snt
from acme import datasets
from acme.specs import EnvironmentSpec
from acme.tf import variable_utils
from acme.utils import counting
from dm_env import specs as dm_specs

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.systems.tf import executors
from mava.systems.tf.maddpg import training
from mava.systems.tf.maddpg.execution import MADDPGFeedForwardExecutor
from mava.wrappers import DetailedTrainerStatistics

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
        shared_weights: boolean indicating whether agents should share weights.
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
    shared_weights: bool = True
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
    max_gradient_norm: Optional[float] = None
    sigma: float = 0.3
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


class MADDPGBuilder:
    """Builder for MADDPG which constructs individual components of the system."""

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

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into.

        Args:
            environment_spec (specs.MAEnvironmentSpec): description of the action and
                observation spaces etc. for each agent in the system.

        Raises:
            NotImplementedError: unknown executor type.

        Returns:
            List[reverb.Table]: a list of data tables for inserting data.
        """

        env_adder_spec = self.convert_discrete_to_bounded(environment_spec)

        # Select adder
        if issubclass(self._executor_fn, executors.FeedForwardExecutor):
            adder_sig = reverb_adders.ParallelNStepTransitionAdder.signature(
                env_adder_spec, self._extra_specs
            )
        elif issubclass(self._executor_fn, executors.RecurrentExecutor):
            adder_sig = reverb_adders.ParallelSequenceAdder.signature(
                env_adder_spec, self._extra_specs
            )
        else:
            raise NotImplementedError("Unknown executor type: ", self._executor_fn)

        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer,
            )

        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=adder_sig,
        )

        return [replay_table]

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
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

        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
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

        # Select adder
        if issubclass(self._executor_fn, executors.FeedForwardExecutor):
            adder = reverb_adders.ParallelNStepTransitionAdder(
                priority_fns=None,
                client=replay_client,
                n_step=self._config.n_step,
                discount=self._config.discount,
            )
        elif issubclass(self._executor_fn, executors.RecurrentExecutor):
            adder = reverb_adders.ParallelSequenceAdder(
                priority_fns=None,
                client=replay_client,
                sequence_length=self._config.sequence_length,
                period=self._config.period,
            )
        else:
            raise NotImplementedError("Unknown executor type: ", self._executor_fn)
        return adder

    def make_executor(
        self,
        policy_networks: Dict[str, snt.Module],
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
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
                update_period=self._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the actor which defines how we take actions.
        return self._executor_fn(
            policy_networks=policy_networks,
            agent_specs=self._config.environment_spec.get_agent_specs(),
            shared_weights=shared_weights,
            variable_client=variable_client,
            adder=adder,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
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

        agents = self._agents
        agent_types = self._agent_types
        shared_weights = self._config.shared_weights
        max_gradient_norm = self._config.max_gradient_norm
        discount = self._config.discount
        target_update_period = self._config.target_update_period
        target_averaging = self._config.target_averaging
        target_update_rate = self._config.target_update_rate

        # trainer args
        trainer_config: Dict[str, Any] = {
            "agents": agents,
            "agent_types": agent_types,
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "observation_networks": networks["observations"],
            "target_policy_networks": networks["target_policies"],
            "target_critic_networks": networks["target_critics"],
            "target_observation_networks": networks["target_observations"],
            "shared_weights": shared_weights,
            "policy_optimizer": self._config.policy_optimizer,
            "critic_optimizer": self._config.critic_optimizer,
            "max_gradient_norm": max_gradient_norm,
            "discount": discount,
            "target_averaging": target_averaging,
            "target_update_period": target_update_period,
            "target_update_rate": target_update_rate,
            "dataset": dataset,
            "counter": counter,
            "logger": logger,
            "checkpoint": self._config.checkpoint,
            "checkpoint_subpath": self._config.checkpoint_subpath,
        }
        if connection_spec:
            trainer_config["connection_spec"] = connection_spec

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(**trainer_config)

        trainer = DetailedTrainerStatistics(  # type: ignore
            trainer, metrics=["policy_loss", "critic_loss"]
        )

        return trainer
