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

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

from mava.systems.ppo.types import PPOTransition
from mava.utils.multistep import calculate_gae


def test_calculate_gae_with_shard_map() -> None:
    """GAE scan carries should vary over the learner device axis."""
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ("learner_devices",))
    num_devices = len(devices)
    shape = (3, num_devices, 2)
    traj_batch = PPOTransition(
        done=jnp.zeros(shape, dtype=bool),
        action=jnp.zeros(shape, dtype=jnp.int32),
        value=jnp.ones(shape),
        reward=jnp.ones(shape),
        log_prob=jnp.zeros(shape),
        obs=jnp.zeros(shape),
    )
    last_val = jnp.ones((num_devices, 2))
    last_done = jnp.zeros((num_devices, 2), dtype=bool)

    def mapped_calculate_gae(
        traj_batch: PPOTransition, last_val: jax.Array, last_done: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return calculate_gae(
            traj_batch,
            last_val,
            last_done,
            gamma=0.99,
            gae_lambda=0.95,
            unroll=1,
            axis_name="learner_devices",
        )

    calculate_sharded_gae = shard_map(
        mapped_calculate_gae,
        mesh=mesh,
        in_specs=(
            PartitionSpec(None, "learner_devices"),
            PartitionSpec("learner_devices"),
            PartitionSpec("learner_devices"),
        ),
        out_specs=(
            PartitionSpec(None, "learner_devices"),
            PartitionSpec(None, "learner_devices"),
        ),
    )

    advantages, targets = calculate_sharded_gae(traj_batch, last_val, last_done)

    np.testing.assert_allclose(advantages[:, 0, 0], [2.79679, 1.921095, 0.99], rtol=1e-5)
    np.testing.assert_allclose(targets, advantages + 1)
