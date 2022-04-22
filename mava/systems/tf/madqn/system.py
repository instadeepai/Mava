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

"""MADQN system implementation."""
import functools
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import acme
import dm_env
import launchpad as lp
import numpy as np
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.tf import utils as tf2_utils
from dm_env import specs

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules.exploration.exploration_scheduling import (
    ConstantScheduler,
)
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf.madqn import builder, training
from mava.systems.tf.madqn.execution import (
    MADQNFeedForwardExecutor,
    sample_new_agent_keys,
)
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.types import EpsilonScheduler
from mava.utils import enums
from mava.utils.loggers import MavaLogger, logger_utils
from mava.utils.sort_utils import sort_str_num
from mava.wrappers import DetailedPerAgentStatistics


class MADQN:
    """MADQN system."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        exploration_scheduler_fn: Union[
            EpsilonScheduler,
            Mapping[str, EpsilonScheduler],
            Mapping[str, Mapping[str, EpsilonScheduler]],
        ],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Type[mava.Trainer] = training.MADQNTrainer,
        executor_fn: Type[core.Executor] = MADQNFeedForwardExecutor,
        num_executors: int = 1,
        trainer_networks: Union[
            Dict[str, List], enums.Trainer
        ] = enums.Trainer.single_trainer,
        network_sampling_setup: Union[
            List, enums.NetworkSampler
        ] = enums.NetworkSampler.fixed_agent_networks,
        fix_sampler: Optional[List] = None,
        net_spec_keys: Dict = {},
        shared_weights: bool = True,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_averaging: bool = False,
        target_update_period: int = 100,
        target_update_rate: Optional[float] = None,
        executor_variable_update_period: int = 1000,
        min_replay_size: int = 1000,
        max_replay_size: int = 100000,
        samples_per_insert: Optional[float] = 32.0,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]] = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        n_step: int = 5,
        sequence_length: int = 20,
        period: int = 10,
        max_gradient_norm: float = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 5,
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
        termination_condition: Optional[Dict[str, int]] = None,
        evaluator_interval: Optional[dict] = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
        seed: Optional[int] = None,
    ):
        """Initialise the system.

        Args:
            environment_factory: function to
                instantiate an environment.
            network_factory: function to instantiate system networks.
            exploration_scheduler_fn: function to schedule
                exploration. e.g. epsilon greedy
            logger_factory: function to
                instantiate a system logger.
            architecture: system architecture, e.g. decentralised or centralised.
            trainer_fn: training type associated with executor and architecture,
                e.g. centralised training.
            executor_fn: executor type, e.g. feedforward or recurrent.
            num_executors: number of executor processes to run in
                parallel.
            trainer_networks: networks each trainer trains on.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
                enums.NetworkSampler settings:
                fixed_agent_networks: Keeps the networks
                used by each agent fixed throughout training.
                random_agent_networks: Creates N network policies, where N is the
                number of agents. Randomly select policies from this sets for each
                agent at the start of a episode. This sampling is done with
                replacement so the same policy can be selected for more than one
                agent for a given episode.
                Custom list: Alternatively one can specify a custom nested list,
                with network keys in, that will be used by the executors at
                the start of each episode to sample networks for each agent.
            fix_sampler: Optional list that can fix the executor sampler to sample
                in a specific way.
            net_spec_keys: Optional network to agent mapping used to get the environment
                specs for each network.
            shared_weights: whether agents should share weights or not.
                When network_sampling_setup are provided the value of shared_weights is
                ignored.
            environment_spec: description of
                the action, observation spaces etc. for each agent in the system.
            discount: discount factor to use for TD updates.
            batch_size: sample batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_averaging: whether to use polyak averaging for
                target network updates.
            target_update_period: number of steps before target
                networks are updated.
            target_update_rate: update rate when using
                averaging.
            executor_variable_update_period: number of steps before
                updating executor variables from the variable source.
            min_replay_size: minimum replay size before updating.
            max_replay_size: maximum replay size.
            samples_per_insert: number of samples to take
                from replay for every insert that is made.
            optimizer: optimizer(s) for updating value networks.
            n_step: number of steps to include prior to boostrapping.
            sequence_length: recurrent sequence rollout length.
            period: Consecutive starting points for overlapping
                rollouts across a sequence.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            checkpoint: whether to checkpoint models.
            checkpoint_subpath: subdirectory specifying where to store
                checkpoints.
            checkpoint_minute_interval: The number of minutes to wait between
                checkpoints.
            logger_config: additional configuration settings for the
                logger factory.
            train_loop_fn: function to instantiate a train loop.
            eval_loop_fn: function to instantiate an evaluation
                loop.
            train_loop_fn_kwargs: possible keyword arguments to send
                to the training loop.
            eval_loop_fn_kwargs: possible keyword arguments to send to
            the evaluation loop.
            termination_condition: An optional terminal condition can be
                provided that stops the program once the condition is
                satisfied. Available options include specifying maximum
                values for trainer_steps, trainer_walltime, evaluator_steps,
                evaluator_episodes, executor_episodes or executor_steps.
                E.g. termination_condition = {'trainer_steps': 100000}.
            evaluator_interval: An optional condition that is used to
                evaluate/test system performance after [evaluator_interval]
                condition has been met. If None, evaluation will
                happen at every timestep.
                E.g. to evaluate a system after every 100 executor episodes,
                evaluator_interval = {"executor_episodes": 100}.
            learning_rate_scheduler_fn: an optional learning rate scheduler for
                the value function optimiser.
            seed: seed for reproducible sampling (used for epsilon
                greedy action selection).

        """

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(
                environment_factory(evaluation=False)  # type: ignore
            )

        # set default logger if no logger provided
        if not logger_factory:
            logger_factory = functools.partial(
                logger_utils.make_logger,
                directory="~/mava",
                to_terminal=True,
                time_delta=10,
            )

        # Setup agent networks and network sampling setup
        agents = sort_str_num(environment_spec.get_agent_ids())
        self._network_sampling_setup = network_sampling_setup

        if type(network_sampling_setup) is not list:
            if network_sampling_setup == enums.NetworkSampler.fixed_agent_networks:
                # if no network_sampling_setup is specified, assign a single network
                # to all agents of the same type if weights are shared
                # else assign seperate networks to each agent
                self._agent_net_keys = {
                    agent: f"network_{agent.split('_')[0]}"
                    if shared_weights
                    else f"network_{agent}"
                    for agent in agents
                }
                self._network_sampling_setup = [
                    [
                        self._agent_net_keys[key]
                        for key in sort_str_num(self._agent_net_keys.keys())
                    ]
                ]
            elif network_sampling_setup == enums.NetworkSampler.random_agent_networks:
                """Create N network policies, where N is the number of agents. Randomly
                select policies from this sets for each agent at the start of a
                episode. This sampling is done with replacement so the same policy
                can be selected for more than one agent for a given episode."""
                if shared_weights:
                    raise ValueError(
                        "Shared weights cannot be used with random policy per agent"
                    )
                self._agent_net_keys = {
                    agents[i]: f"network_{i}" for i in range(len(agents))
                }
                self._network_sampling_setup = [
                    [
                        [self._agent_net_keys[key]]
                        for key in sort_str_num(self._agent_net_keys.keys())
                    ]
                ]
            else:
                raise ValueError(
                    "network_sampling_setup must be a dict or fixed_agent_networks"
                )

        else:
            # if a dictionary is provided, use network_sampling_setup to determine setup
            _, self._agent_net_keys = sample_new_agent_keys(
                agents,
                self._network_sampling_setup,  # type: ignore
                fix_sampler=fix_sampler,
            )
        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(self._network_sampling_setup[0])  # type: ignore
        assert len(environment_spec.get_agent_ids()) == len(self._agent_net_keys.keys())

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(self._agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(self._network_sampling_setup)):  # type: ignore
            assert len(self._network_sampling_setup[i]) == sample_length  # type: ignore

        # Get all the unique agent network keys
        all_samples = []
        for sample in self._network_sampling_setup:  # type: ignore
            all_samples.extend(sample)
        unique_net_keys = list(sort_str_num(list(set(all_samples))))

        # Create mapping from ints to networks
        net_keys_to_ids = {net_key: i for i, net_key in enumerate(unique_net_keys)}

        # Setup trainer_networks
        if type(trainer_networks) is not dict:
            if trainer_networks == enums.Trainer.single_trainer:
                self._trainer_networks = {"trainer": unique_net_keys}
            elif trainer_networks == enums.Trainer.one_trainer_per_network:
                self._trainer_networks = {
                    f"trainer_{i}": [unique_net_keys[i]]
                    for i in range(len(unique_net_keys))
                }
            else:
                raise ValueError(
                    "trainer_networks does not support this enums setting."
                )
        else:
            self._trainer_networks = trainer_networks  # type: ignore

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in self._trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        if net_spec_keys:
            self._net_spec_keys = net_spec_keys
        else:
            self._net_spec_keys = {}
            for i in range(len(unique_net_keys)):
                self._net_spec_keys[unique_net_keys[i]] = agents[i % len(agents)]

        # Setup table_network_config
        table_network_config = {}
        for trainer_key in self._trainer_networks.keys():
            most_matches = 0
            trainer_nets = self._trainer_networks[trainer_key]
            for sample in self._network_sampling_setup:  # type: ignore
                matches = 0
                for entry in sample:
                    if entry in trainer_nets:
                        matches += 1
                if most_matches < matches:
                    matches = most_matches
                    table_network_config[trainer_key] = sample

        # Setup epsilon schedules
        # If we receive a single schedule, we use that for all agents.
        self._num_exectors = num_executors
        if not isinstance(exploration_scheduler_fn, dict):
            self._exploration_scheduler_fn: Dict = {}
            for executor_id in range(self._num_exectors):
                executor = f"executor_{executor_id}"
                self._exploration_scheduler_fn[executor] = {}
                for agent in environment_spec.get_agent_ids():
                    self._exploration_scheduler_fn[executor][
                        agent
                    ] = exploration_scheduler_fn
        else:
            # Using an executor level config
            if all(key.startswith("executor") for key in exploration_scheduler_fn):
                self._exploration_scheduler_fn = exploration_scheduler_fn

            # Using an agent level config - assume all executor have the same config.
            elif all(key.startswith("agent") for key in exploration_scheduler_fn):
                for executor_id in range(self._num_exectors):
                    self._exploration_scheduler_fn[
                        executor_id
                    ] = exploration_scheduler_fn

            else:
                raise ValueError(
                    "Unknown exploration_scheduler_fn, please pass a callable or a dict"
                    + " with an agent or executor"
                    + f" level config: {exploration_scheduler_fn}"
                )

        self._table_network_config = table_network_config
        self._architecture = architecture
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._logger_factory = logger_factory
        self._environment_spec = environment_spec
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint = checkpoint
        self._logger_config = logger_config
        self._train_loop_fn = train_loop_fn
        self._train_loop_fn_kwargs = train_loop_fn_kwargs
        self._eval_loop_fn = eval_loop_fn
        self._eval_loop_fn_kwargs = eval_loop_fn_kwargs
        self._evaluator_interval = evaluator_interval
        self._seed = seed

        extra_specs = {}
        if issubclass(executor_fn, executors.RecurrentExecutor):
            extra_specs = self._get_extra_specs()

        int_spec = specs.DiscreteArray(len(unique_net_keys))
        agents = environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        extra_specs.update(net_spec)

        self._builder = builder.MADQNBuilder(
            builder.MADQNConfig(
                environment_spec=environment_spec,
                agent_net_keys=self._agent_net_keys,
                trainer_networks=self._trainer_networks,
                table_network_config=table_network_config,
                num_executors=num_executors,
                network_sampling_setup=self._network_sampling_setup,  # type: ignore
                fix_sampler=fix_sampler,
                net_spec_keys=self._net_spec_keys,
                net_keys_to_ids=net_keys_to_ids,
                unique_net_keys=unique_net_keys,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_averaging=target_averaging,
                target_update_period=target_update_period,
                target_update_rate=target_update_rate,
                executor_variable_update_period=executor_variable_update_period,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                sequence_length=sequence_length,
                period=period,
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                optimizer=optimizer,
                checkpoint_subpath=checkpoint_subpath,
                checkpoint_minute_interval=checkpoint_minute_interval,
                termination_condition=termination_condition,
                evaluator_interval=evaluator_interval,
                learning_rate_scheduler_fn=learning_rate_scheduler_fn,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
        )

    def _get_extra_specs(self) -> Any:
        """Helper to establish specs for extra information.

        Returns:
            Dictionary containing extra specs
        """

        agents = self._environment_spec.get_agent_ids()
        core_state_specs = {}
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )
        for agent in agents:
            agent_net_key = self._agent_net_keys[agent]
            core_state_specs[agent] = (
                tf2_utils.squeeze_batch_dim(
                    networks["values"][agent_net_key].initial_state(1)
                ),
            )
        return {"core_states": core_state_specs, "zero_padding_mask": np.array(1)}

    def replay(self) -> Any:
        """Step counter.

        Args:
            checkpoint: whether to checkpoint the counter.

        Returns:
            step counter object.
        """
        return self._builder.make_replay_tables(self._environment_spec)

    def create_system(
        self,
    ) -> Dict:
        """Initialise the system variables from the network factory."""
        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # architecture args
        architecture_config = {
            "environment_spec": self._environment_spec,
            "observation_networks": networks["observations"],
            "value_networks": networks["values"],
            "action_selectors": networks["action_selectors"],
            "agent_net_keys": self._agent_net_keys,
        }

        system = self._architecture(**architecture_config)
        networks = system.create_system()

        return networks

    def variable_server(self) -> MavaVariableSource:
        """Create the variable server."""
        # Create the system
        networks = self.create_system()
        return self._builder.make_variable_server(networks)

    def executor(
        self,
        executor_id: str,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
    ) -> mava.ParallelEnvironmentLoop:
        """System executor.

        Args:
            executor_id: id to identify the executor process for logging purposes.
            replay: replay data table to push data to.
            variable_source: variable server for updating
                network variables.

        Returns:
            mava.ParallelEnvironmentLoop: environment-executor loop instance.
        """

        # Create the system
        networks = self.create_system()

        # Create the executor.
        executor = self._builder.make_executor(
            networks=networks,
            exploration_schedules=self._exploration_scheduler_fn[
                f"executor_{executor_id}"
            ],
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
            evaluator=False,
            seed=self._seed,
        )

        # TODO (Arnu): figure out why factory function are giving type errors
        # Create the environment.
        environment = self._environment_factory(evaluation=False)  # type: ignore

        # Create executor logger
        executor_logger_config = {}
        if self._logger_config and "executor" in self._logger_config:
            executor_logger_config = self._logger_config["executor"]
        exec_logger = self._logger_factory(  # type: ignore
            f"executor_{executor_id}", **executor_logger_config
        )

        # Create the loop to connect environment and executor.
        train_loop = self._train_loop_fn(
            environment,
            executor,
            logger=exec_logger,
            **self._train_loop_fn_kwargs,
        )

        train_loop = DetailedPerAgentStatistics(train_loop)

        return train_loop

    def evaluator(
        self,
        variable_source: acme.VariableSource,
    ) -> Any:
        """System evaluator.

        Args:
            variable_source: variable server for updating
                network variables.
            logger: logger object.

        Returns:
            environment-executor evaluation loop instance for evaluating the
                performance of a system.
        """

        # Create the system
        networks = self.create_system()

        # Create the agent.
        executor = self._builder.make_executor(
            evaluator=True,
            networks=networks,
            exploration_schedules={
                agent: ConstantScheduler(epsilon=0.0)
                for agent in self._environment_spec.get_agent_ids()
            },
            variable_source=variable_source,
            seed=self._seed,
        )

        # Make the environment.
        environment = self._environment_factory(evaluation=True)  # type: ignore

        # Create logger and counter.
        evaluator_logger_config = {}
        if self._logger_config and "evaluator" in self._logger_config:
            evaluator_logger_config = self._logger_config["evaluator"]
        eval_logger = self._logger_factory(  # type: ignore
            "evaluator", **evaluator_logger_config
        )

        # Create the run loop and return it.
        # Create the loop to connect environment and executor.
        eval_loop = self._eval_loop_fn(
            environment,
            executor,
            logger=eval_logger,
            **self._eval_loop_fn_kwargs,
        )

        eval_loop = DetailedPerAgentStatistics(eval_loop)
        return eval_loop

    def trainer(
        self,
        trainer_id: str,
        replay: reverb.Client,
        variable_source: MavaVariableSource,
    ) -> mava.core.Trainer:
        """System trainer.

        Args:
            trainer_id: Id of the trainer being created.
            replay: replay data table to pull data from.
            variable_source: variable server for updating
                network variables.

        Returns:
            system trainer.
        """
        # create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            trainer_id, **trainer_logger_config
        )

        # Create the system
        networks = self.create_system()

        dataset = self._builder.make_dataset_iterator(replay, trainer_id)

        return self._builder.make_trainer(
            networks=networks,
            trainer_networks=self._trainer_networks[trainer_id],
            trainer_table_entry=self._table_network_config[trainer_id],
            dataset=dataset,
            logger=trainer_logger,
            variable_source=variable_source,
        )

    def build(self, name: str = "madqn") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name: system name.

        Returns:
            graph program for distributed system training.
        """
        program = lp.Program(name=name)

        with program.group("replay"):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group("variable_server"):
            variable_server = program.add_node(lp.CourierNode(self.variable_server))

        with program.group("trainer"):
            # Add executors which pull round-robin from our variable sources.
            for trainer_id in self._trainer_networks.keys():
                program.add_node(
                    lp.CourierNode(self.trainer, trainer_id, replay, variable_server)
                )

        with program.group("evaluator"):
            program.add_node(lp.CourierNode(self.evaluator, variable_server))

        with program.group("executor"):
            # Add executors which pull round-robin from our variable sources.
            for executor_id in range(self._num_exectors):
                program.add_node(
                    lp.CourierNode(self.executor, executor_id, replay, variable_server)
                )

        return program
