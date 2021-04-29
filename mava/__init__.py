"""mava is a framework for multi-agent reinforcement learning."""

# Expose specs and types modules.
from acme import types as acme_types

# Internal core import.
from acme.core import Saveable, VariableSource, Worker

from mava import specs, types, utils

# Make __version__ accessible.
from mava._metadata import __version__

# Expose core interfaces.
from mava.core import Executor, Trainer

# Expose the environment loop.
from mava.environment_loop import ParallelEnvironmentLoop, SequentialEnvironmentLoop
from mava.specs import MAEnvironmentSpec

# Internal environment_loop import.

# Mava loves you more. ;)
