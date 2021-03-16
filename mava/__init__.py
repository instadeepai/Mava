"""mava is a framework for multi-agent reinforcement learning."""

# Expose specs and types modules.
from acme import types

# Internal core import.
from acme.core import Saveable, VariableSource, Worker

from mava import specs

# Make __version__ accessible.
from mava._metadata import __version__

# Expose core interfaces.
from mava.core import Executor, Trainer
from mava.specs import SystemSpec

# Expose the environment loop.
# from mava.environment_loop import EnvironmentLoop

# Internal environment_loop import.


# Mava loves you more. ;)
