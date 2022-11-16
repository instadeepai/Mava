# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Utility function for building and launching launchpad programs."""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional

import launchpad as lp
import psutil
from absl import flags, logging
from acme.utils import counting
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mava.core_jax import SystemParameterServer
from mava.utils.training_utils import non_blocking_sleep

FLAGS = flags.FLAGS


def to_device(program_nodes: List, nodes_on_gpu: List = ["trainer"]) -> Dict:
    """Specifies which nodes should run on gpu.

    If nodes_on_gpu is an empty list, this returns a cpu only config.

    Args:
        program_nodes (List): nodes in lp program.
        nodes_on_gpu (List, optional): nodes to run on gpu. Defaults to ["trainer"].

    Returns:
        Dict: dict with cpu only lp config.
    """
    return {
        node: PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)})
        if (node not in nodes_on_gpu)
        else []
        for node in program_nodes
    }


def partial_kwargs(function: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Return a partial function application by overriding default keywords.

    This function is equivalent to `functools.partial(function, **kwargs)` but
    will raise a `ValueError` when called if either the given keyword arguments
    are not defined by `function` or if they do not have defaults.
    This is useful as a way to define a factory function with default parameters
    and then to override them in a safe way.

    Args:
      function: the base function before partial application.
      **kwargs: keyword argument overrides.

    Returns:
      A function.
    """
    # Try to get the argspec of our function which we'll use to get which keywords
    # have defaults.
    argspec = inspect.getfullargspec(function)

    # Figure out which keywords have defaults.
    if argspec.defaults is None:
        defaults = []
    else:
        defaults = argspec.args[-len(argspec.defaults) :]

    # Find any keys not given as defaults by the function.
    unknown_kwargs = set(kwargs.keys()).difference(defaults)

    # Raise an error
    if unknown_kwargs:
        error_string = "Cannot override unknown or non-default kwargs: {}"
        raise ValueError(error_string.format(", ".join(unknown_kwargs)))

    return functools.partial(function, **kwargs)


def termination_fn(
    parameter_server: SystemParameterServer,
) -> None:
    """Terminate the process

    Args:
        parameter_server: SystemParameterServer in order to get main pid
    """
    if parameter_server.store.manager_pid:
        # parent_pid: the pid of the main thread process
        parent_pid = parameter_server.store.manager_pid
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            child.kill()
    else:
        lp.stop()


class StepsLimiter:
    def __init__(
        self,
        counter: counting.Counter,
        max_steps: Optional[int],
        steps_key: str = "executor_steps",
    ):
        """Process that terminates an experiment when `max_steps` is reached."""
        self._counter = counter
        self._max_steps = max_steps
        self._steps_key = steps_key

    def run(self) -> None:
        """Run steps limiter to terminate an experiment when max_steps is reached."""

        logging.info(
            "StepsLimiter: Starting with max_steps = %d (%s)",
            self._max_steps,
            self._steps_key,
        )
        while True:
            # Update the counts.
            counts = self._counter.get_counts()
            num_steps = counts.get(self._steps_key, 0)

            logging.info("StepsLimiter: Reached %d recorded steps", num_steps)

            if num_steps > self._max_steps:
                logging.info(
                    "StepsLimiter: Max steps of %d was reached, terminating",
                    self._max_steps,
                )
                lp.stop()

            # Don't spam the counter.
            non_blocking_sleep(10)
