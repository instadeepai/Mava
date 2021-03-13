"""mava is a framework for multi-agent reinforcement learning."""

# Expose specs and types modules.
from mava import specs
from mava import types

# Make __version__ accessible.
from mava._metadata import __version__

# Expose core interfaces.
from mava.core import Executor

# Internal core import.
from mava.core import Trainer
from mava.core import Saveable
from mava.core import VariableSource
from mava.core import Worker

# Expose the environment loop.
from mava.environment_loop import EnvironmentLoop

# Internal environment_loop import.

from mava.specs import SystemSpec

# Mava loves you more. ;)