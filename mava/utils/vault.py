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

import json
import os
from datetime import datetime
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import tensorstore as ts
from chex import Array
from etils import epath
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState

# CURRENT LIMITATIONS / TODO LIST
# - Metadata write and read
# - Vault multiplier ???
# - Anakin -> extra minibatch dim...
# - Only works when fbx_state.is_full is False (i.e. can't handle wraparound)
# - Better async stuff
# - Only tested with flat buffers

DRIVER = "file://"
METADATA_FILE = "metadata.json"
TIME_AXIS_MAX_LENGTH = int(10e12)
VERSION = 0.1


class Vault:
    def __init__(
        self,
        init_fbx_state: TrajectoryBufferState,
        vault_name: str,
        rel_dir: str = "vaults",
        vault_uid: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:

        vault_str = vault_uid if vault_uid else datetime.now().strftime("%Y%m%d%H%M%S")
        self._base_path = os.path.join(os.getcwd(), rel_dir, vault_name, vault_str)

        # We use epath for metadata
        metadata_path = epath.Path(os.path.join(self._base_path, METADATA_FILE))

        # Check if the vault exists, otherwise create the necessary dirs and files
        base_path_exists = os.path.exists(self._base_path)
        if base_path_exists:
            self._metadata = json.loads(metadata_path.read_text())
            # Ensure minor versions match
            assert (self._metadata["version"] // 1) == (VERSION // 1)
        else:
            # Create the necessary dirs for the vault
            os.makedirs(self._base_path)
            self._metadata = {
                "version": VERSION,
                **(metadata or {}),  # Allow user to save extra metadata
            }
            metadata_path.write_text(json.dumps(self._metadata))

        # Keep a data store for the vault index
        self._vault_index_ds = ts.open(
            self._get_base_spec("vault_index"),
            dtype=jnp.int32,
            shape=(1,),
            create=not base_path_exists,
        ).result()
        self.vault_index = int(self._vault_index_ds.read().result()[0])

        # Each leaf of the fbx_state.experience is a data store
        self._all_ds = jax.tree_util.tree_map_with_path(
            lambda path, x: self._init_leaf(
                name=jax.tree_util.keystr(path),  # Use the path as the name
                leaf=x,
                create_checkpoint=not base_path_exists,
            ),
            init_fbx_state.experience,
        )

        # Just store one timestep for the structure
        self._fbx_sample_experience = jax.tree_map(
            lambda x: x[:, 0:1, ...],
            init_fbx_state.experience,
        )
        self._last_received_fbx_index = 0

    def _get_base_spec(self, name: str) -> dict:
        return {
            "driver": "zarr",
            "kvstore": {
                "driver": "ocdbt",
                "base": f"{DRIVER}{self._base_path}",  # TODO: does this work on other systems?
                "path": name,
            },
        }

    def _init_leaf(self, name: str, leaf: Array, create_checkpoint: bool = False) -> ts.TensorStore:
        spec = self._get_base_spec(name)
        leaf_ds = ts.open(
            spec,
            dtype=leaf.dtype if create_checkpoint else None,
            shape=(
                leaf.shape[0],  # Batch dim
                TIME_AXIS_MAX_LENGTH,  # Time dim
                *leaf.shape[2:],  # Experience dim
            )
            if create_checkpoint
            else None,
            create=create_checkpoint,
        ).result()
        return leaf_ds

    def _write_leaf(
        self,
        source_leaf: Array,
        dest_leaf: ts.TensorStore,
        source_interval: Tuple[Optional[int], Optional[int]],
        dest_interval: Tuple[Optional[int], Optional[int]],
    ) -> None:
        dest_leaf[:, slice(*dest_interval), ...].write(
            source_leaf[:, slice(*source_interval), ...],
        ).result()

    def write(
        self,
        fbx_state: TrajectoryBufferState,
        source_interval: Tuple[Optional[int], Optional[int]] = (None, None),
        dest_interval: Tuple[Optional[int], Optional[int]] = (None, None),
    ) -> None:
        # TODO: more than one current_index if B > 1
        fbx_current_index = int(fbx_state.current_index)

        if source_interval == (None, None):
            source_interval = (self._last_received_fbx_index, fbx_current_index)
        write_length = source_interval[1] - source_interval[0]  # type: ignore
        if write_length == 0:
            return  # Nothing to write

        if dest_interval == (None, None):
            dest_interval = (self.vault_index, self.vault_index + write_length)

        # If user has provided a dest_interval, check that it's the correct length
        assert write_length == (dest_interval[1] - dest_interval[0])  # type: ignore

        # Write to each ds
        jax.tree_util.tree_map(
            lambda x, ds: self._write_leaf(
                source_leaf=x,
                dest_leaf=ds,
                source_interval=source_interval,
                dest_interval=dest_interval,
            ),
            fbx_state.experience,  # x = experience
            self._all_ds,  # ds = data stores
        )

        # Update vault index, and write this to the ds too
        self.vault_index += write_length
        self._vault_index_ds.write(self.vault_index).result()

        # Keep track of the last fbx buffer idx received
        self._last_received_fbx_index = fbx_current_index

    def _read_leaf(
        self,
        read_leaf: ts.TensorStore,
        read_interval: Tuple[Optional[int], Optional[int]],
    ) -> Array:
        return read_leaf[:, slice(*read_interval), ...].read().result()

    def read(
        self, read_interval: Tuple[Optional[int], Optional[int]] = (None, None)
    ) -> Array:  # TODO typing
        if read_interval == (None, None):
            read_interval = (0, self.vault_index)  # Read all that has been written

        return jax.tree_util.tree_map(
            lambda _, ds: self._read_leaf(
                read_leaf=ds,
                read_interval=read_interval,
            ),
            self._fbx_sample_experience,  # just for structure
            self._all_ds,  # data stores
        )

    def get_buffer(
        self, size: int, key: Array, starting_index: Optional[int] = None
    ) -> TrajectoryBufferState:
        assert size <= self.vault_index
        if starting_index is None:
            starting_index = int(
                jax.random.randint(
                    key=key,
                    shape=(),
                    minval=0,
                    maxval=self.vault_index - size,
                )
            )
        return TrajectoryBufferState(
            experience=self.read((starting_index, starting_index + size)),
            current_index=starting_index + size,
            is_full=True,
        )
