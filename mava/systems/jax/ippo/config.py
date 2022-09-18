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

"""Default hyperparameters for IPPO system."""
from dataclasses import dataclass

from mava.components.jax import executing


@dataclass
class IPPODefaultConfig:
    sample_batch_size: int = 512
    sequence_length: int = 20
    period: int = 10
    use_next_extras: bool = False


# The components that a needed for IPPO with a recurrent policy.
recurrent_policy_components = [
    executing.RecurrentExecutorSelectAction,
    executing.RecurrentExecutorObserve,
]
