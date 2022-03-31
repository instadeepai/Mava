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

"""Jax MAPPO system."""

from mava.specs import DesignSpec
from mava.systems.jax import System
from mava.systems.jax.mappo import EXECUTOR_SPEC, TRAINER_SPEC


class PPOSystem(System):
    def design(self) -> DesignSpec:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        executor = EXECUTOR_SPEC.get()
        trainer = TRAINER_SPEC.get()
        system_design = DesignSpec(
            **executor,
            **trainer,
        )
        return system_design
