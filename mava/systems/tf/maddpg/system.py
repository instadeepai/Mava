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

"""MADDPG system implementation."""

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import acme
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from dm_env import specs

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import (
    DecentralisedQValueActorCritic,
    DecentralisedValueActorCritic,
)
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf.maddpg import builder, training
from mava.systems.tf.maddpg.execution import (
    MADDPGFeedForwardExecutor,
    sample_new_agent_keys,
)
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.utils.loggers import MavaLogger, logger_utils
from mava.utils.sort_utils import sort_str_num
from mava.wrappers import DetailedPerAgentStatistics


class MADDPG:
    """MADDPG system."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[
            DecentralisedQValueActorCritic
        ] = DecentralisedQValueActorCritic,
        trainer_fn: Union[
            Type[training.MADDPGBaseTrainer],
            Type[training.MADDPGBaseRecurrentTrainer],
        ] = training.MADDPGDecentralisedTrainer,
        executor_fn: Type[core.Executor] = MADDPGFeedForwardExecutor,
        num_executors: int = 1,
        trainer_networks: Dict[str, List] = {},
        executor_samples: List = [],
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
        policy_optimizer: Union[
            snt.Optimizer, Dict[str, snt.Optimizer]
        ] = snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        n_step: int = 5,
        sequence_length: int = 20,
        period: int = 20,
        bootstrap_n: int = 10,
        sigma: float = 0.3,
        max_gradient_norm: float = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
        connection_spec: Callable[[Dict[str, List[str]]], Dict[str, List[str]]] = None,
    ):
        """Initialise the system
        Args:
            environment_factory (Callable[[bool], dm_env.Environment]): function to
                instantiate an environment.
            network_factory (Callable[[acme_specs.BoundedArray],
                Dict[str, snt.Module]]): function to instantiate system networks.
            logger_factory (Callable[[str], MavaLogger], optional): function to
                instantiate a system logger. Defaults to None.
            architecture (Type[ DecentralisedQValueActorCritic ], optional):
                system architecture, e.g. decentralised or centralised. Defaults to
                DecentralisedQValueActorCritic.
            trainer_fn (Union[ Type[training.MADDPGBaseTrainer],
                Type[training.MADDPGBaseRecurrentTrainer], ], optional): training type
                associated with executor and architecture, e.g. centralised training.
                Defaults to training.MADDPGDecentralisedTrainer.
            executor_fn (Type[core.Executor], optional): executor type, e.g.
                feedforward or recurrent. Defaults to MADDPGFeedForwardExecutor.
            num_executors (int, optional): number of executor processes to run in
                parallel. Defaults to 1.
            num_caches (int, optional): number of trainer node caches. Defaults to 0.
            environment_spec (mava_specs.MAEnvironmentSpec, optional): description of
                the action, observation spaces etc. for each agent in the system.
                Defaults to None.
            trainer_networks (Dict[str, List[snt.Module]], optional): networks each
                trainer trains on. Defaults to {}.
            executor_samples (List, optional): List of networks that are randomly
                sampled from by the executors at the start of an environment run.
                Defaults to [].
            shared_weights (bool, optional): whether agents should share weights or not.
                When executor_samples are provided the value of shared_weights is
                ignored. Defaults to True.
            discount (float, optional): discount factor to use for TD updates. Defaults
                to 0.99.
            batch_size (int, optional): sample batch size for updates. Defaults to 256.
            prefetch_size (int, optional): size to prefetch from replay. Defaults to 4.
            target_averaging (bool, optional): whether to use polyak averaging for
                target network updates. Defaults to False.
            target_update_period (int, optional): number of steps before target
                networks are updated. Defaults to 100.
            target_update_rate (Optional[float], optional): update rate when using
                averaging. Defaults toNone.
            executor_variable_update_period (int, optional): number of steps before
                updating executor variables from the variable source. Defaults to 1000.
            min_replay_size (int, optional): minimum replay size before updating.
                Defaults to 1000.
            max_replay_size (int, optional): maximum replay size. Defaults to 1000000.
            samples_per_insert (Optional[float], optional): number of samples to take
                from replay for every insert that is made. Defaults to 32.0.
            policy_optimizer (Union[ snt.Optimizer, Dict[str, snt.Optimizer] ],
                optional): optimizer(s) for updating policy networks. Defaults to
                snt.optimizers.Adam(learning_rate=1e-4).
            critic_optimizer (snt.Optimizer, optional): optimizer for updating critic
                networks. Defaults to snt.optimizers.Adam(learning_rate=1e-4).
            n_step (int, optional): number of steps to include prior to boostrapping.
                Defaults to 5.
            sequence_length (int, optional): recurrent sequence rollout length. Defaults
                to 20.
            period (int, optional): Consecutive starting points for overlapping
                rollouts across a sequence. Defaults to 20.
            bootstrap_n (int, optional): Used to determine the spacing between
                q_value/value estimation for bootstrapping. Should be less
                than sequence_length.
            sigma (float, optional): Gaussian sigma parameter. Defaults to 0.3.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            max_executor_steps (int, optional): maximum number of steps and executor
                can in an episode. Defaults to None.
            checkpoint (bool, optional): whether to checkpoint models. Defaults to
                False.
            checkpoint_subpath (str, optional): subdirectory specifying where to store
                checkpoints. Defaults to "~/mava/".
            logger_config (Dict, optional): additional configuration settings for the
                logger factory. Defaults to {}.
            train_loop_fn (Callable, optional): function to instantiate a train loop.
                Defaults to ParallelEnvironmentLoop.
            eval_loop_fn (Callable, optional): function to instantiate an evaluation
                loop. Defaults to ParallelEnvironmentLoop.
            train_loop_fn_kwargs (Dict, optional): possible keyword arguments to send
                to the training loop. Defaults to {}.
            eval_loop_fn_kwargs (Dict, optional): possible keyword arguments to send to
                the evaluation loop. Defaults to {}.
            connection_spec (Callable[[Dict[str, List[str]]], Dict[str, List[str]]],
                optional): network topology specification for networked system
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

        # Setup agent networks and executor sampler
        agents = sort_str_num(environment_spec.get_agent_ids())
        self._executor_samples = executor_samples
        if not executor_samples:
            # if no executor samples provided, use shared_weights to determine setup
            self._agent_net_keys = {
                agent: agent.split("_")[0] if shared_weights else agent
                for agent in agents
            }
            self._executor_samples = [
                [
                    self._agent_net_keys[key]
                    for key in sort_str_num(self._agent_net_keys.keys())
                ]
            ]
        else:
            # if executor samples provided, use executor_samples to determine setup
            _, self._agent_net_keys = sample_new_agent_keys(
                agents,
                self._executor_samples,
            )

        # Check that the environment and agent_net_keys has the same amount of agents
        # TODO (dries): Try using self._executor_samples = [['agent]].
        # This might accelerate training as experience gets batched instead of training
        # sequentially.
        sample_length = len(self._executor_samples[0])
        assert len(environment_spec.get_agent_ids()) == len(self._agent_net_keys.keys())

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(self._agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(self._executor_samples)):
            assert len(self._executor_samples[i]) == sample_length

        # Get all the unique agent network keys
        all_samples = []
        for sample in self._executor_samples:
            all_samples.extend(sample)
        unique_net_keys = sort_str_num(list(set(all_samples)))

        # Create mapping from ints to networks
        net_to_ints = {net_key: i for i, net_key in enumerate(unique_net_keys)}

        # Setup trainer_networks
        if not trainer_networks:
            self._trainer_networks = {"trainer_0": list(unique_net_keys)}
        else:
            self._trainer_networks = trainer_networks

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in self._trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        self._net_spec_keys = {}
        for i in range(len(unique_net_keys)):
            self._net_spec_keys[unique_net_keys[i]] = agents[i % len(agents)]

        # Setup table_network_config
        table_network_config = {}
        for t_id in range(len(self._trainer_networks.keys())):
            most_matches = 0
            trainer_nets = self._trainer_networks[f"trainer_{t_id}"]
            for sample in self._executor_samples:
                matches = 0
                for entry in sample:
                    if entry in trainer_nets:
                        matches += 1
                if most_matches < matches:
                    matches = most_matches
                    table_network_config[f"trainer_{t_id}"] = sample

        self._table_network_config = table_network_config
        self._architecture = architecture
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._logger_factory = logger_factory
        self._environment_spec = environment_spec
        self._num_exectors = num_executors
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint = checkpoint
        self._logger_config = logger_config
        self._train_loop_fn = train_loop_fn
        self._train_loop_fn_kwargs = train_loop_fn_kwargs
        self._eval_loop_fn = eval_loop_fn
        self._eval_loop_fn_kwargs = eval_loop_fn_kwargs

        if connection_spec:
            self._connection_spec = connection_spec(  # type: ignore
                environment_spec.get_agents_by_type()
            )
        else:
            self._connection_spec = None  # type: ignore

        extra_specs = {}
        if issubclass(executor_fn, executors.RecurrentExecutor):
            extra_specs = self._get_extra_specs()

        int_spec = specs.DiscreteArray(len(unique_net_keys))
        agents = environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        extra_specs.update(net_spec)

        self._builder = builder.MADDPGBuilder(
            builder.MADDPGConfig(
                environment_spec=environment_spec,
                agent_net_keys=self._agent_net_keys,
                trainer_networks=self._trainer_networks,
                table_network_config=table_network_config,
                num_executors=num_executors,
                executor_samples=self._executor_samples,
                net_to_ints=net_to_ints,
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
                bootstrap_n=bootstrap_n,
                sigma=sigma,
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                checkpoint_subpath=checkpoint_subpath,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
        )

    def _get_extra_specs(self) -> Any:
        """helper to establish specs for extra information
        Returns:
            Dict[str, Any]: dictionary containing extra specs
        """

        agents = self._environment_spec.get_agent_ids()
        core_state_specs = {}
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
            net_spec_keys=self._net_spec_keys,
        )
        for agent in agents:
            agent_net_key = self._agent_net_keys[agent]
            core_state_specs[agent] = (
                tf2_utils.squeeze_batch_dim(
                    networks["policies"][agent_net_key].initial_state(1)
                ),
            )
        return {"core_states": core_state_specs}

    def replay(self) -> Any:
        """Step counter
        Args:
            checkpoint (bool): whether to checkpoint the counter.
        Returns:
            Any: step counter object.
        """
        return self._builder.make_replay_tables(self._environment_spec)

    def create_system(
        self,
    ) -> Tuple[Dict[str, Dict[str, snt.Module]], Dict[str, Dict[str, snt.Module]]]:
        """Initialise the system variables from the network factory."""
        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
            net_spec_keys=self._net_spec_keys,
        )

        # Create system architecture with target networks.
        adder_env_spec = self._builder.convert_discrete_to_bounded(
            self._environment_spec
        )

        # architecture args
        architecture_config = {
            "environment_spec": adder_env_spec,
            "observation_networks": networks["observations"],
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "agent_net_keys": self._agent_net_keys,
        }

        # net_spec_keys is only implemented for the Decentralised architectures
        if (
            self._architecture == DecentralisedValueActorCritic
            or self._architecture == DecentralisedQValueActorCritic
        ):
            architecture_config["net_spec_keys"] = self._net_spec_keys

        # TODO (dries): Can net_spec_keys and network_spec be used as
        # the same thing? Can we use use one of those two instead of both.

        if self._connection_spec:
            architecture_config["network_spec"] = self._connection_spec
        system = self._architecture(**architecture_config)
        networks = system.create_system()
        behaviour_networks = system.create_behaviour_policy()
        return behaviour_networks, networks

    def variable_server(self) -> MavaVariableSource:
        """Create the variable server."""
        # Create the system
        _, networks = self.create_system()
        return self._builder.make_variable_server(networks)

    def executor(
        self,
        executor_id: str,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
    ) -> mava.ParallelEnvironmentLoop:
        """System executor
        Args:
            executor_id (str): id to identify the executor process for logging purposes.
            replay (reverb.Client): replay data table to push data to.
            variable_source (acme.VariableSource): variable server for updating
                network variables.
            counter (counting.Counter): step counter object.
        Returns:
            mava.ParallelEnvironmentLoop: environment-executor loop instance.
        """
        # Create the system
        behaviour_policy_networks, networks = self.create_system()

        # Create the executor.
        executor = self._builder.make_executor(
            executor_id=executor_id,
            networks=networks,
            policy_networks=behaviour_policy_networks,
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
        )

        # TODO (Arnu): figure out why factory function are giving type errors
        # Create the environment.
        environment = self._environment_factory(evaluation=False)  # type: ignore

        # Create executor logger
        executor_logger_config = {}
        if self._logger_config and "executor" in self._logger_config:
            executor_logger_config = self._logger_config["executor"]
        exec_logger = self._logger_factory(  # type: ignore
            executor_id, **executor_logger_config
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
        logger: loggers.Logger = None,
    ) -> Any:
        """System evaluator (an executor process not connected to a dataset)
        Args:
            variable_source (acme.VariableSource): variable server for updating
                network variables.
            counter (counting.Counter): step counter object.
            logger (loggers.Logger, optional): logger object. Defaults to None.
        Returns:
            Any: environment-executor evaluation loop instance for evaluating the
                performance of a system.
        """

        # Create the system
        behaviour_policy_networks, networks = self.create_system()

        # Create the agent.
        executor = self._builder.make_executor(
            executor_id="evaluator",
            networks=networks,
            policy_networks=behaviour_policy_networks,
            variable_source=variable_source,
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
        # counter: counting.Counter,
    ) -> mava.core.Trainer:
        """System trainer
        Args:
            replay (reverb.Client): replay data table to pull data from.
            counter (counting.Counter): step counter object.
        Returns:
            mava.core.Trainer: system trainer.
        """

        # create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            trainer_id, **trainer_logger_config
        )

        # Create the system
        _, networks = self.create_system()

        dataset = self._builder.make_dataset_iterator(replay, trainer_id)

        return self._builder.make_trainer(
            trainer_id=trainer_id,
            networks=networks,
            trainer_networks=self._trainer_networks[trainer_id],
            trainer_table_entry=self._table_network_config[trainer_id],
            dataset=dataset,
            logger=trainer_logger,
            connection_spec=self._connection_spec,
            variable_source=variable_source,
        )

    def build(self, name: str = "maddpg") -> Any:
        """Build the distributed system as a graph program.
        Args:
            name (str, optional): system name. Defaults to "maddpg".
        Returns:
            Any: graph program for distributed system training.
        """
        program = lp.Program(name=name)

        with program.group("replay"):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group("variable_server"):
            variable_server = program.add_node(lp.CourierNode(self.variable_server))

        with program.group("trainer"):
            # Add executors which pull round-robin from our variable sources.
            for trainer_id in range(len(self._trainer_networks.keys())):
                program.add_node(
                    lp.CourierNode(
                        self.trainer, f"trainer_{trainer_id}", replay, variable_server
                    )
                )

        with program.group("evaluator"):
            program.add_node(lp.CourierNode(self.evaluator, variable_server))

        with program.group("executor"):
            # Add executors which pull round-robin from our variable sources.
            for executor_id in range(self._num_exectors):
                program.add_node(
                    lp.CourierNode(
                        self.executor,
                        f"executor_{executor_id}",
                        replay,
                        variable_server,
                    )
                )

        return program
