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

"""MAPPO system implementation."""

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
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActorCritic
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf.mappo import builder, execution, training
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.utils import enums
from mava.utils.loggers import MavaLogger, logger_utils
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num
from mava.wrappers import DetailedPerAgentStatistics


class MAPPO:
    """MAPPO system."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[
            DecentralisedValueActorCritic
        ] = DecentralisedValueActorCritic,
        trainer_fn: Type[training.MAPPOTrainer] = training.MAPPOTrainer,
        executor_fn: Type[
            Union[execution.MAPPOFeedForwardExecutor, execution.MAPPORecurrentExecutor]
        ] = execution.MAPPOFeedForwardExecutor,
        num_executors: int = 1,
        trainer_networks: Union[
            Dict[str, List], enums.Trainer
        ] = enums.Trainer.single_trainer,
        network_sampling_setup: Union[
            List, enums.NetworkSampler
        ] = enums.NetworkSampler.fixed_agent_networks,
        fix_sampler: Optional[List] = None,
        net_spec_keys: Dict = {},
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        executor_variable_update_period: int = 100,
        policy_optimizer: Union[
            snt.Optimizer, Dict[str, snt.Optimizer]
        ] = snt.optimizers.Adam(learning_rate=5e-4),
        critic_optimizer: Optional[snt.Optimizer] = snt.optimizers.Adam(
            learning_rate=5e-4
        ),
        discount: float = 0.99,
        lambda_gae: float = 0.95,
        clipping_epsilon: float = 0.2,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        max_gradient_norm: Optional[float] = 0.5,
        max_queue_size: Optional[int] = None,
        batch_size: int = 512,
        minibatch_size: int = None,
        num_epochs: int = 5,
        sequence_length: int = 10,
        sequence_period: Optional[int] = None,
        max_executor_steps: int = None,
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
        normalize_advantage: bool = False,
    ):
        """Initialise the system

        Args:
            environment_factory: function to
                instantiate an environment.
            network_factory : function to instantiate system networks.
            logger_factory : function to
                instantiate a system logger. Defaults to None.
            architecture : system
                architecture, e.g. decentralised or centralised. Defaults to
                DecentralisedValueActorCritic.
            trainer_fn : training type
                associated with executor and architecture, e.g. centralised training.
                Defaults to training.MAPPOTrainer.
            executor_fn : executor type, e.g. feedforward
                or recurrent. Defaults to execution.MAPPOFeedForwardExecutor.
            num_executors : number of executor processes to run in
                parallel. Defaults to 1.
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
            num_caches : number of trainer node caches. Defaults to 0.
            environment_spec : description of
                the action, observation spaces etc. for each agent in the system.
                Defaults to None.
            shared_weights : whether agents should share weights or not.
                When agent_net_keys are provided the value of shared_weights is ignored.
                Defaults to True.
            agent_net_keys: : specifies what network each agent uses.
                Defaults to {}.
            executor_variable_update_period : number of steps before
                updating executor variables from the variable source. Defaults to 100.
            policy_optimizer : optimizer(s) for updating policy networks.
                Defaults to snt.optimizers.Adam(learning_rate=5e-4).
            critic_optimizer : optimizer for updating critic
                networks. This is not used if using single optim.
            discount : discount factor to use for TD updates. Defaults
                to 0.99.
            lambda_gae : scalar determining the mix of bootstrapping
                vs further accumulation of multi-step returns at each timestep.
                Defaults to 0.99.
            clipping_epsilon : Hyper-parameter for clipping in the
                policy objective. Defaults to 0.2.
            entropy_cost : contribution of entropy regularization to
                the total loss. Defaults to 0.01.
            baseline_cost : contribution of the value loss to the
                total loss. Defaults to 0.5.
            max_gradient_norm: value to specify the maximum clipping value for the
            gradient norm during optimization.
            max_queue_size : maximum number of items in the queue.
                Should be larger than batch size.
            batch_size: sample batch size for updates.
                Defaults to 512. Minibatches are sampled from this data.
            minibatch_size: size of minibatch that is sampled
                from the training batch. Minibatches are used for each gradient step.
            num_epochs: number of epochs every training step.
                Recommendation as per https://arxiv.org/pdf/2103.01955.pdf, "15
                epochs for easy tasks,and 10 or 5 epochs for difficult tasks."
            sequence_length: sequence rollout length. Defaults
                to 10.
            sequence_period: consecutive starting points for
                overlapping rollouts across a sequence. Defaults to sequence length -1.
            max_executor_steps: maximum number of steps and executor
                can in an episode. Defaults to None.
            checkpoint : whether to checkpoint models. Defaults to
                False.
            checkpoint_subpath: subdirectory specifying where to store
                checkpoints. Defaults to "~/mava/".
            checkpoint_minute_interval: The number of minutes to wait between
                checkpoints.
            logger_config: additional configuration settings for the
                logger factory. Defaults to {}.
            train_loop_fn: function to instantiate a train loop.
                Defaults to ParallelEnvironmentLoop.
            eval_loop_fn: function to instantiate an evaluation
                loop. Defaults to ParallelEnvironmentLoop.
            train_loop_fn_kwargs: possible keyword arguments to send
                to the training loop. Defaults to {}.
            eval_loop_fn_kwargs: possible keyword arguments to send to
                the evaluation loop. Defaults to {}.
            learning_rate_scheduler_fn: dict with two functions/classes (one for the
                policy and one for the critic optimizer), that takes in a trainer
                step t and returns the current learning rate,
                e.g. {"policy": policy_lr_schedule ,"critic": critic_lr_schedule}.
                See
                examples/debugging/simple_spread/feedforward/decentralised/run_maddpg_lr_schedule.py
                for an example.
            evaluator_interval: An optional condition that is used to
                evaluate/test system performance after [evaluator_interval]
                condition has been met. If None, evaluation will
                happen at every timestep.
                E.g. to evaluate a system after every 100 executor episodes,
                evaluator_interval = {"executor_episodes": 100}.
            normalize_advantage: whether to normalize the advantage estimate. This can
                hurt peformance when shared weights are used.
        """
        # minibatch size defaults to train batch size
        if minibatch_size:
            self._minibatch_size = minibatch_size
        else:
            self._minibatch_size = batch_size

        assert batch_size % self._minibatch_size == 0, (
            "batch_size must be divisible by minibatch_size."
            + f"Got batch_size={batch_size},"
            + f"minibatch_size={self._minibatch_size}"
        )

        if max_queue_size:
            self._max_queue_size = max_queue_size
        else:
            # NOTE: Based on https://github.com/deepmind/acme/blob/6bf350df1d9dd16cd85217908ec9f47553278976/acme/agents/jax/ppo/builder.py#L75 # noqa: E501
            extra_capacity_to_avoid_single_machine_deadlocks = 1000
            self._max_queue_size = (
                batch_size + extra_capacity_to_avoid_single_machine_deadlocks
            )

        assert self._max_queue_size > batch_size, (
            "Max queue size should be larger than batch size."
            + f"Got max_queue_size={self._max_queue_size},"
            + f"batch_size={batch_size}"
        )

        # An extra step is used for bootstrapping when computing advantages.
        self._sequence_length = sequence_length + 1

        if sequence_period:
            self._sequence_period = sequence_period
        else:
            self._sequence_period = self._sequence_length - 1

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
                # if no network_sampling_setup is fixed, use shared_weights to
                # determine setup
                self._agent_net_keys = {
                    agent: "network_0" if shared_weights else f"network_{i}"
                    for i, agent in enumerate(agents)
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
        self._net_spec_keys = net_spec_keys
        if not net_spec_keys:
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

        extra_specs = {}
        if issubclass(executor_fn, executors.RecurrentExecutor):
            extra_specs = self._get_extra_specs()

        int_spec = specs.DiscreteArray(len(unique_net_keys))
        agents = environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        extra_specs.update(net_spec)
        self._evaluator_interval = evaluator_interval
        self._builder = builder.MAPPOBuilder(
            config=builder.MAPPOConfig(
                environment_spec=environment_spec,
                agent_net_keys=self._agent_net_keys,
                executor_variable_update_period=executor_variable_update_period,
                discount=discount,
                lambda_gae=lambda_gae,
                clipping_epsilon=clipping_epsilon,
                entropy_cost=entropy_cost,
                baseline_cost=baseline_cost,
                max_gradient_norm=max_gradient_norm,
                max_queue_size=self._max_queue_size,
                batch_size=batch_size,
                minibatch_size=self._minibatch_size,
                num_epochs=num_epochs,
                sequence_length=self._sequence_length,
                sequence_period=self._sequence_period,
                checkpoint=checkpoint,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                checkpoint_subpath=checkpoint_subpath,
                checkpoint_minute_interval=checkpoint_minute_interval,
                trainer_networks=self._trainer_networks,
                table_network_config=table_network_config,
                network_sampling_setup=self._network_sampling_setup,  # type: ignore
                fix_sampler=fix_sampler,
                net_spec_keys=self._net_spec_keys,
                net_keys_to_ids=net_keys_to_ids,
                unique_net_keys=unique_net_keys,
                termination_condition=termination_condition,
                evaluator_interval=evaluator_interval,
                learning_rate_scheduler_fn=learning_rate_scheduler_fn,
                normalize_advantage=normalize_advantage,
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
        )
        for agent in agents:
            net_keys = self._agent_net_keys[agent]
            core_state_specs[agent] = (
                tf2_utils.squeeze_batch_dim(
                    networks["policies"][net_keys].initial_state(1)
                ),
            )
        return {"core_states": core_state_specs}

    def replay(self) -> Any:
        """Replay data storage.

        Returns:
            Any: replay data table built according the environment specification.
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

        # architecture args
        architecture_config = {
            "environment_spec": self._environment_spec,
            "observation_networks": networks["observations"],
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "agent_net_keys": self._agent_net_keys,
        }

        # net_spec_keys is only implemented for the Decentralised architectures
        if self._architecture == DecentralisedValueActorCritic:
            architecture_config["net_spec_keys"] = self._net_spec_keys

        # TODO (dries): Can net_spec_keys and network_spec be used as
        # the same thing? Can we use use one of those two instead of both.

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
            executor_id: id to identify the executor process for logging purposes.
            replay: replay data table to push data to.
            variable_source: variable server for updating
                network variables.
        Returns:
            mava.ParallelEnvironmentLoop: environment-executor loop instance.
        """

        # Create the system
        behaviour_policy_networks, networks = self.create_system()

        # Create the executor.
        executor = self._builder.make_executor(
            networks=networks,
            policy_networks=behaviour_policy_networks,
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
            evaluator=False,
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
        logger: loggers.Logger = None,
    ) -> Any:
        """System evaluator (an executor process not connected to a dataset)
        Args:
            variable_source: variable server for updating
                network variables.
            logger: logger object.
        Returns:
            environment-executor evaluation loop instance for evaluating the
                performance of a system.
        """

        # Create the system
        behaviour_policy_networks, networks = self.create_system()

        # Create the agent.
        executor = self._builder.make_executor(
            # executor_id="evaluator",
            networks=networks,
            policy_networks=behaviour_policy_networks,
            variable_source=variable_source,
            evaluator=True,
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
        """System trainer
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
        _, networks = self.create_system()

        dataset = self._builder.make_dataset_iterator(replay, trainer_id)

        return self._builder.make_trainer(
            networks=networks,
            trainer_networks=self._trainer_networks[trainer_id],
            trainer_table_entry=self._table_network_config[trainer_id],
            dataset=dataset,
            logger=trainer_logger,
            variable_source=variable_source,
        )

    def build(self, name: str = "maddpg") -> Any:
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
