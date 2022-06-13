from types import SimpleNamespace

# from typing import Callable, List, Tuple
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import pytest

from mava.components.jax.training.losses import (
    MAPGTrustRegionClippingLossConfig,
    MAPGWithTrustRegionClippingLoss,
)
from mava.systems.jax.trainer import Trainer


@pytest.fixture
def mock_trainer() -> Trainer:
    """Creates mock trainer fixture"""

    mock_trainer = Trainer(config=SimpleNamespace())
    mock_trainer.store.grad_fn = None

    return mock_trainer


# @pytest.fixture
def mapg_trust_region_clipping_loss_config() -> MAPGTrustRegionClippingLossConfig:
    """Creates an MAPG loss config fixture with trust region and clipping"""

    test_mapg_config = MAPGTrustRegionClippingLossConfig()

    return test_mapg_config


@pytest.fixture
def mapg_trust_region_clipping_loss() -> MAPGWithTrustRegionClippingLoss:
    """Creates an MAPG loss fixture with trust region and clipping"""

    test_mapg = MAPGWithTrustRegionClippingLoss(
        config=MAPGTrustRegionClippingLossConfig()
    )

    return test_mapg


def test_mapg_creation(
    mock_trainer: Trainer,
    mapg_trust_region_clipping_loss: MAPGWithTrustRegionClippingLoss,
) -> None:
    """Test whether mapg function is successfully created"""

    mapg_trust_region_clipping_loss.on_training_loss_fns(trainer=mock_trainer)
    assert hasattr(mock_trainer.store, "grad_fn")
    assert isinstance(mock_trainer.store.grad_fn, Callable)  # type:ignore


def test_mapg_config_creation() -> None:
    """Test whether mapg loss config variables are of correct type"""

    mapg_config = mapg_trust_region_clipping_loss_config()

    assert isinstance(mapg_config.clipping_epsilon, float)
    assert isinstance(mapg_config.clip_value, bool)
    assert isinstance(mapg_config.entropy_cost, float)
    assert isinstance(mapg_config.value_cost, float)



