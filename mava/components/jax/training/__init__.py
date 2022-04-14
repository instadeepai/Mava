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

"""Trainer components for Mava systems."""
from mava.components.jax.training.advantage_estimation import GAE
from mava.components.jax.training.base import Batch, Loss, Step, TrainingState, Utility
from mava.components.jax.training.losses import (
    MAMCTSLoss,
    MAPGWithTrustRegionClippingLoss,
)
from mava.components.jax.training.model_updating import (
    MAMCTSEpochUpdate,
    MAMCTSMinibatchUpdate,
    MAPGEpochUpdate,
    MAPGMinibatchUpdate,
)
from mava.components.jax.training.n_step_bootstrapped_returns import (
    NStepBootStrappedReturns,
)
from mava.components.jax.training.step import (
    DefaultStep,
    MAMCTSStep,
    MAPGWithTrustRegionStep,
)
from mava.components.jax.training.trainer import TrainerInit
