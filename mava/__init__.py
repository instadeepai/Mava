"""mava is a framework for multi-agent reinforcement learning."""

# Expose specs and types modules.
from mava import specs
from acme import types

# Make __version__ accessible.
from mava._metadata import __version__

# Expose core interfaces.
from mava.core import Executor
from mava.core import Trainer

# Internal core import.
from acme.core import Saveable
from acme.core import VariableSource
from acme.core import Worker

# Expose the environment loop.
# from mava.environment_loop import EnvironmentLoop

# Internal environment_loop import.

from mava.specs import SystemSpec

# Mava loves you more. ;)