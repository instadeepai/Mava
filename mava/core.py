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


"""Core Mava interfaces."""

import abc
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import dm_env
import numpy as np
import reverb
import sonnet as snt
from acme import core as acme_core
from acme import types

# import mava
from mava import adders
from mava.systems.launcher import Launcher
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource

T = TypeVar("T")


class SystemExecutor(abc.ABC):
    """Abstract system executor object."""

    @abc.abstractmethod
    def _policy(
        self, agent: str, observation: types.NestedTensor
    ) -> types.NestedTensor:
        """Agent specific policy function"""

    @abc.abstractmethod
    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> Union[types.NestedArray, Tuple[types.NestedArray, types.NestedArray]]:
        """select an action for a single agent in the system"""

    @abc.abstractmethod
    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record first observed timestep from the environment"""

    @abc.abstractmethod
    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record observed timestep from the environment"""

    @abc.abstractmethod
    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Union[
        Dict[str, types.NestedArray],
        Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]],
    ]:
        """select the actions for all agents in the system"""

    @abc.abstractmethod
    def update(self, wait: bool = False) -> None:
        """update executor variables"""


class SystemTrainer(abc.ABC):
    """Abstract system trainer object."""

    @abc.abstractmethod
    def _update_target_networks(self) -> None:
        """Sync the target network parameters with the latest online network
        parameters"""

    @abc.abstractmethod
    def _transform_observations(
        self, obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray]
    ) -> Any:
        """Transform the observations using the observation networks of each agent."""

    @abc.abstractmethod
    def _get_feed(
        self,
        transition: Dict[str, Dict[str, np.ndarray]],
        agent: str,
    ) -> Any:
        """get data to feed to the agent networks"""

    @abc.abstractmethod
    def _step(self) -> Dict:
        """Trainer forward and backward passes."""

    @abc.abstractmethod
    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass"""

    @abc.abstractmethod
    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

    @abc.abstractmethod
    def step(self) -> None:
        """trainer step to update the parameters of the agents in the system"""


class SystemBuilder(abc.ABC):
    """Abstract system builder object."""

    @abc.abstractmethod
    def tables(
        self,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""

    @abc.abstractmethod
    def dataset(
        self,
        replay_client: reverb.Client,
        table_name: str,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for training/updating the system."""

    @abc.abstractmethod
    def adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment."""

    @abc.abstractmethod
    def system(
        self,
    ) -> Tuple[Dict[str, Dict[str, snt.Module]], Dict[str, Dict[str, snt.Module]]]:
        """[summary]"""

    @abc.abstractmethod
    def variable_server(
        self,
    ) -> MavaVariableSource:
        """Create the variable server."""

    @abc.abstractmethod
    def executor(
        self,
        executor_id: str,
        replay_client: reverb.Client,
        variable_source: MavaVariableSource,
    ) -> acme_core.Worker:
        """[summary]"""

    @abc.abstractmethod
    def evaluator(
        self,
        variable_source: MavaVariableSource,
    ) -> Any:
        """[summary]"""
        iterator = range(num_steps) if num_steps is not None else itertools.count()
        for _ in iterator:
            self.step()
            self.after_trainer_step()

    # TODO(Arnu/Kale-ab): find a more suitable way to do this using callbacks
    def after_trainer_step(self) -> None:
        """Function that gets executed after every trainer step."""
        pass

    @abc.abstractmethod
    def trainer(
        self,
        trainer_id: str,
        replay_client: reverb.Client,
        variable_source: MavaVariableSource,
    ) -> SystemTrainer:
        """[summary]"""

    @abc.abstractmethod
    def build(
        self,
        program: Launcher,
    ) -> Any:
        """[summary]"""


class System(abc.ABC):
    """Abstract system object."""

    @abc.abstractmethod
    def build(self, name: str) -> None:
        """[summary]

        Args:
            name (str): [description]
        """

    @abc.abstractmethod
    def distribute(
        self,
        num_executors: int,
        nodes_on_gpu: List[str],
    ) -> None:
        """[summary]

        Args:
            num_executors (int): [description]
            nodes_on_gpu (List[str]): [description]
        """

    @abc.abstractmethod
    def launch(self):
        """[summary]"""