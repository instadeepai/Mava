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

import queue
import threading
import time
from typing import Any, List, Union

import jax
from colorama import Fore, Style

from mava.systems.ppo.types import Params
from mava.utils.sebulba.pipelines import OffPolicyPipeline, Pipeline


class RecordTimeTo:
    """Context manager to record the runtime in a `with` block"""

    def __init__(self, to: Any):
        self.to = to

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, *args: Any) -> None:
        end = time.monotonic()
        self.to.append(end - self.start)


class ParamsSource(threading.Thread):
    """A `ParamSource` is a component that allows networks params to be passed from a
    `Learner` component to `Actor` components.
    """

    def __init__(self, init_value: Params, device: jax.Device):
        super().__init__(name=f"ParamsSource-{device.id}")
        self.value: Params = jax.device_put(init_value, device)
        self.device = device
        self.new_value: queue.Queue = queue.Queue()
        self._should_stop = False

    def run(self) -> None:
        """This function is responsible for updating the value of the `ParamSource` when a new value
        is available.
        """
        while not self._should_stop:
            try:
                waiting = self.new_value.get(block=True, timeout=1)
                self.value = jax.device_put(waiting, self.device)
            except queue.Empty:
                continue

    def update(self, new_params: Params) -> None:
        """Update the value of the `ParamSource` with a new value.

        Args:
            new_params: The new value to update the `ParamSource` with.
        """
        self.new_value.put(new_params)

    def get(self) -> Params:
        """Get the current value of the `ParamSource`."""
        return self.value

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._should_stop = True


def stop_sebulba(
    actors_stop_event: threading.Event,
    pipe: Union[Pipeline, OffPolicyPipeline],
    params_sources: List[ParamsSource],
    actor_threads: List[threading.Thread],
) -> None:
    actors_stop_event.set()
    pipe.clear()  # We clear the pipeline before stopping the actor threads to avoid deadlock
    print(f"{Fore.RED}{Style.BRIGHT}Pipe cleared{Style.RESET_ALL}")
    print(f"{Fore.RED}{Style.BRIGHT}Stopping actor threads...{Style.RESET_ALL}")
    for actor in actor_threads:
        actor.join()
        print(f"{Fore.RED}{Style.BRIGHT}{actor.name} stopped{Style.RESET_ALL}")
    print(f"{Fore.RED}{Style.BRIGHT}Stopping pipeline...{Style.RESET_ALL}")
    pipe.stop()
    pipe.join()
    print(f"{Fore.RED}{Style.BRIGHT}Stopping params sources...{Style.RESET_ALL}")
    for params_source in params_sources:
        params_source.stop()
        params_source.join()
    print(f"{Fore.RED}{Style.BRIGHT}All threads stopped...{Style.RESET_ALL}")
