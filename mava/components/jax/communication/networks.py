from typing import Any, Dict, List, Sequence

from mava import specs as mava_specs


def make_default_gcn(
    environment_spec: mava_specs.MAEnvironmentSpec,
    rng_key: List[int],
    update_head_sizes: Sequence[int] = (128, 128),
) -> Dict[str, Any]:
    """Create default GCN to use for communication."""
    return {}
