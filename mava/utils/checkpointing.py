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


import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Tuple

import orbax.checkpoint as ocp
from etils import epath
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from orbax.checkpoint.checkpoint_managers import AnyPreservationPolicy, BestN, LatestN


def best_fn(metrics: Dict[str, float]) -> float:
    """Default function to determine performance of checkpoint. Uses `metrics['episode_return']`."""
    return metrics["episode_return"]


def make_checkpointer(
    cfg: DictConfig, best_fn: Callable[[Dict[str, float]], float] = best_fn
) -> ocp.CheckpointManager:
    """Initializes and returns an Orbax CheckpointManager based on the config.

    This function configures a CheckpointManager for saving model checkpoints.
    It constructs a save directory based on the provided configuration, sets up
    a preservation policy to keep both the latest and the best checkpoints,
    and embeds the experiment's configuration as metadata within the checkpoint
    directory.

    Args:
        cfg (DictConfig): The Hydra configuration object. It should contain
            settings for the checkpointer path (`cfg.checkpointer.save.path`),
            the system name (`cfg.logger.system_name`), a unique ID
            (`cfg.checkpointer.save.uid`), and preservation policy settings
            (`cfg.checkpointer.save.preservation_policy`).
        best_fn (Callable[[Dict[str, float]], float]): A function that takes a
            metrics dictionary and returns a float value used to determine the
            "best" checkpoint. Defaults to a function that uses
            `metrics["episode_return"]`.

    Returns:
        ocp.CheckpointManager: An initialized Orbax CheckpointManager ready for saving.

    Raises:
        AssertionError: If `cfg.checkpointer.save.use` is False

    Example Usage:
        ```
        from orbax.checkpoint.args import Composite, StandardSave

        chkptr = make_checkpointer(cfg)
        chkptr.save(
            step,
            args=Composite(
                params=StandardSave(learner_state.params),
                opt_states=StandardSave(learner_state.opt_states),
            ),
        )
        ```
    """
    assert cfg.checkpointer.save.use, f"Can't checkpoint if {cfg.checkpointer.save.use=}"
    # If uid is None then make it date-time
    uid = cfg.checkpointer.save.uid
    if uid is None:
        uid = datetime.now().strftime("%Y%m%d%H%M%S")
        print(f"cfg.checkpointer.save.uid not found, creating {uid=}")
    # if path is relative then make it absolute
    path = epath.Path(cfg.checkpointer.save.path, cfg.logger.system_name, uid).resolve()

    # TODO: check that this works with mngr.latest_step() and mngr.best_step()
    # Determines which checkpoints to keep
    best_to_keep = BestN(
        get_metric_fn=best_fn,
        n=cfg.checkpointer.save.preservation_policy.num_best,
        keep_checkpoints_without_metrics=False,
    )
    latest_to_keep = LatestN(n=cfg.checkpointer.save.preservation_policy.num_latest)
    preservation_policy = AnyPreservationPolicy([best_to_keep, latest_to_keep])

    options = ocp.CheckpointManagerOptions(create=True, preservation_policy=preservation_policy)
    mngr = ocp.CheckpointManager(
        path,
        options=options,
        metadata=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
    )

    return mngr


def load_checkpoint(cfg: DictConfig) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Loads a checkpoint and its associated configuration from a path specified in config.

    This function restores a saved training state. It locates the checkpoint
    directory using the provided configuration, determines which specific step
    to load ('latest', 'best', or a specific integer), and restores the
    checkpoint data. It also returns the original configuration that was
    saved with the checkpoint.

    Args:
        cfg (DictConfig): The Hydra configuration object for loading. It must
            specify the load path (`cfg.checkpointer.load.path`), the system
            name (`cfg.logger.system_name`), the unique ID of the run
            (`cfg.checkpointer.load.uid`), and which step to load
            (`cfg.checkpointer.load.step`).

    Returns:
        Tuple[Any, dict]: A tuple containing:
            - The restored checkpoint data (e.g., model parameters, optimizer state).
            - The configuration dictionary that was saved with the checkpoint.

    Raises:
        AssertionError: If `cfg.checkpointer.load.use` is False or if
            `cfg.checkpointer.load.uid` is not provided.
        ValueError: If `cfg.checkpointer.load.step` is not a positive integer,
            'latest', or 'best'.
    """
    assert cfg.checkpointer.load.use, f"Can't checkpoint if {cfg.checkpointer.load.use=}"
    assert cfg.checkpointer.load.uid is not None, f"Can't checkpoint: {cfg.checkpointer.load.uid=}"

    path = epath.Path(cfg.checkpointer.load.path, cfg.logger.system_name, cfg.checkpointer.load.uid)
    path = path.resolve()
    _step = cfg.checkpointer.load.step

    mngr = ocp.CheckpointManager(path)

    # Warn if the config.logger.system_name doesn't match the loaded system's config
    load_cfg = mngr.metadata().custom_metadata
    load_system_name = load_cfg["logger"]["system_name"]  # type: ignore
    if load_system_name != cfg.logger.system_name:
        warn = (
            f"Loading system ({load_system_name}) "
            f"doesn't match current system ({cfg.logger.system_name})"
        )
        warnings.warn(warn, stacklevel=1)

    # Get step from config
    if isinstance(_step, int) and _step > 0:
        step = _step
    elif _step == "latest":
        step = mngr.latest_step()
    elif _step == "best":
        step = mngr.best_step()
    else:
        err = f"Unrecognised {cfg.checkpointer.load.step=}. Expected int > 0 or 'latest' or 'best'"
        raise ValueError(err)

    return mngr.restore(step), load_cfg  # type: ignore
