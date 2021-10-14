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


"""Core Mava interfaces.
This file specifies and documents the notions of `Executor` and `Trainer`
similar to the `Actor` and `Learner` in Acme.
"""

import abc
import itertools
from typing import Dict, Generic, Optional, Sequence, Tuple, TypeVar, Union

import acme
import dm_env
from acme import types

T = TypeVar("T")

import abc
from typing import Any, Dict, Iterator, List, Optional

import reverb
import sonnet as snt


from mava import adders, types
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource


class SystemExecutor(abc.ABC):
    """[summary]"""

    @abc.abstractmethod
    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> Union[types.NestedArray, Tuple[types.NestedArray, types.NestedArray]]:
        """select an action for a single agent in the system

        Args:
            agent (str): agent id.
            observation (types.NestedArray): observation tensor received from the
                environment.

        Returns:
            Union[types.NestedArray, Tuple[types.NestedArray, types.NestedArray]]:
                agent action.
        """

    @abc.abstractmethod
    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record first observed timestep from the environment

        Args:
            timestep (dm_env.TimeStep): data emitted by an environment at first step of
                interaction.
            extras (Dict[str, types.NestedArray], optional): possible extra information
                to record during the first step. Defaults to {}.
        """

    @abc.abstractmethod
    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record observed timestep from the environment

        Args:
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """

    @abc.abstractmethod
    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Union[
        Dict[str, types.NestedArray],
        Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]],
    ]:
        """select the actions for all agents in the system

        Args:
            observations (Dict[str, types.NestedArray]): agent observations from the
                environment.

        Returns:
            Union[ Dict[str, types.NestedArray], Tuple[Dict[str, types.NestedArray],
                Dict[str, types.NestedArray]], ]: actions for all agents in the system.
        """

    @abc.abstractmethod
    def update(self, wait: bool = False) -> None:
        """update executor variables

        Args
            wait (bool, optional): whether to stall the executor's request for new
                variables. Defaults to False.
        """


class VariableSource(abc.ABC):
    """Abstract source of variables.
    Objects which implement this interface provide a source of variables, returned
    as a collection of (nested) numpy arrays. Generally this will be used to
    provide variables to some learned policy/etc.
    """

    @abc.abstractmethod
    def get_variables(
        self, names: Sequence[str]
    ) -> Dict[str, Dict[str, types.NestedArray]]:
        """Return the named variables as a collection of (nested) numpy arrays.
        Args:
        names: args where each name is a string identifying a predefined subset of
            the variables.
        Returns:
        A list of (nested) numpy arrays `variables` such that `variables[i]`
        corresponds to the collection named by `names[i]`.
        """


class Worker(abc.ABC):
    """An interface for (potentially) distributed workers."""

    @abc.abstractmethod
    def run(self) -> None:
        """Runs the worker."""


class Saveable(abc.ABC, Generic[T]):
    """An interface for saveable objects."""

    @abc.abstractmethod
    def save(self) -> T:
        """Returns the state from the object to be saved."""

    @abc.abstractmethod
    def restore(self, state: T) -> None:
        """Given the state, restores the object."""


class SystemTrainer(VariableSource, Worker, Saveable):
    """Abstract learner object.
    This corresponds to an object which implements a learning loop. A single step
    of learning should be implemented via the `step` method and this step
    is generally interacted with via the `run` method which runs update
    continuously.
    All objects implementing this interface should also be able to take in an
    external dataset (see acme.datasets) and run updates using data from this
    dataset. This can be accomplished by explicitly running `learner.step()`
    inside a for/while loop or by using the `learner.run()` convenience function.
    Data will be read from this dataset asynchronously and this is primarily
    useful when the dataset is filled by an external process.
    """

    @abc.abstractmethod
    def step(self) -> None:
        """Perform an update step of the learner's parameters."""

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the update loop; typically an infinite loop which calls step."""

        iterator = range(num_steps) if num_steps is not None else itertools.count()

        for _ in iterator:
            self.step()

    def save(self) -> T:
        raise NotImplementedError('Method "save" is not implemented.')

    def restore(self, state: T) -> None:
        raise NotImplementedError('Method "restore" is not implemented.')


class SystemBuilder(abc.ABC):
    """Builder for systems which constructs individual components of the
    system."""

    @abc.abstractmethod
    def tables(
        self,
    ) -> List[reverb.Table]:
        """ "Create tables to insert data into.
        Args:
            environment_spec (specs.MAEnvironmentSpec): description of the action and
                observation spaces etc. for each agent in the system.
        Returns:
            List[reverb.Table]: a list of data tables for inserting data.
        """

    @abc.abstractmethod
    def dataset(
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

    @abc.abstractmethod
    def adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
            replay_client (reverb.Client): Reverb Client which points to the
                replay server.
        Returns:
            Optional[adders.ParallelAdder]: adder which sends data to a replay buffer.
        """

    @abc.abstractmethod
    def system(
        self,
    ) -> Tuple[Dict[str, Dict[str, snt.Module]], Dict[str, Dict[str, snt.Module]]]:
        """[summary]

        Returns:
            Tuple[Dict[str, Dict[str, snt.Module]], Dict[str, Dict[str, snt.Module]]]: [description]
        """

    @abc.abstractmethod
    def make_variable_server(
        self,
    ) -> MavaVariableSource:
        """Create the variable server.
        Returns:
            variable_source (MavaVariableSource): A Mava variable source object.
        """

    @abc.abstractmethod
    def executor(
        self,
        executor_id: str,
        replay_client: reverb.Client,
        variable_source: acme.VariableSource,
    ) -> mava.ParallelEnvironmentLoop:
        """[summary]

        Args:
            executor_id (str): [description]
            replay_client (reverb.Client): [description]
            variable_source (acme.VariableSource): [description]

        Returns:
            mava.ParallelEnvironmentLoop: [description]
        """

    @abc.abstractmethod
    def evaluator(
        self,
        variable_source: acme.VariableSource,
    ) -> Any:
        """[summary]

        Args:
            variable_source (acme.VariableSource): [description]

        Returns:
            Any: [description]
        """

    @abc.abstractmethod
    def trainer(
        self,
        trainer_id: str,
        replay_client: reverb.Client,
        variable_source: MavaVariableSource,
    ) -> mava.core.Trainer:
        """[summary]

        Args:
            trainer_id (str): [description]
            replay_client (reverb.Client): [description]
            variable_source (MavaVariableSource): [description]

        Returns:
            mava.core.Trainer: [description]
        """

    @abc.abstractmethod
    def distributor(self) -> Any:
        """[summary]

        Returns:
            Any: [description]
        """