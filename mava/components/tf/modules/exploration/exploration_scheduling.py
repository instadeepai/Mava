# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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
class LinearExplorationScheduler:
    def __init__(self, epsilon_min: float = 0.05, epsilon_decay: float = 1e-4):
        """
        Decays epsilon linearly to zero.
        """
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._epsilon = 1.0

    def decrement_epsilon(self) -> None:
        if self._epsilon == self._epsilon_min:
            return

        self._epsilon -= self._epsilon_decay
        if self._epsilon < self._epsilon_min:
            self._epsilon = self._epsilon_min

    def get_epsilon(self) -> float:
        return self._epsilon

    def reset_epsilon(self) -> None:
        self._epsilon = 1.0


class ExponentialExplorationScheduler(LinearExplorationScheduler):
    def __init__(
        self, logdir: str, epsilon_min: float = 0.05, epsilon_decay: float = 1e-4
    ):
        """
        Decays epsilon exponentially to zero.
        """
        super().__init__(epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)

    # TODO (Claude) implement exponential decay.
    def decrement_epsilon(self) -> None:
        raise NotImplementedError
