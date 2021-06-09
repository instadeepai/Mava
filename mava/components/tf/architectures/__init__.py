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

"""MARL system architectural components."""

from mava.components.tf.architectures.base import (
    BaseActorCritic,
    BaseArchitecture,
    BasePolicyArchitecture,
)
from mava.components.tf.architectures.centralised import (
    CentralisedPolicyActor,
    CentralisedQValueActorCritic,
    CentralisedQValueCritic,
    CentralisedValueCritic,
)
from mava.components.tf.architectures.decentralised import (
    DecentralisedPolicyActor,
    DecentralisedQValueActorCritic,
    DecentralisedValueActor,
    DecentralisedValueActorCritic,
)
from mava.components.tf.architectures.networked import (
    NetworkedPolicyActor,
    NetworkedQValueCritic,
)
from mava.components.tf.architectures.state_based import (
    StateBasedPolicyActor,
    StateBasedQValueActorCritic,
    StateBasedQValueCritic,
)
