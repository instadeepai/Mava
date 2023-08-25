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

import timeit
from typing import Any, Optional

from colorama import Fore, Style


class TimeIt:
    """Context manager for timing execution.

    Usage:
        Wrap a block of code with the `TimeIt` context manager to measure its execution time.
        Optionally, you can provide the number of environment steps (`environment_steps`)
        to calculate the steps per second (SPS) metric.

    Note:
        This implementation is a generic context manager for timing execution using Python's
        `timeit` module.
        For the original implementation, please refer to the following link:
        (https://colab.research.google.com/drive/1974D-qP17fd5mLxy6QZv-ic4yxlPJp-G?usp=sharing)
    """

    def __init__(self, tag: str, environment_steps: Optional[int] = None) -> None:
        """Initialise the context manager."""
        self.tag = tag
        self.environment_steps = environment_steps

    def __enter__(self) -> "TimeIt":
        """Start the timer."""
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args: Any) -> None:
        """Print the elapsed time."""
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.environment_steps:
            msg += ", SPS=%.2e" % (self.environment_steps / self.elapsed_secs)
        print(f"{Fore.YELLOW}{Style.BRIGHT}{msg}{Style.RESET_ALL}")
