from typing import List

import dm_env
from acme.utils import counting, loggers

import mava
from mava.environment_loop import SequentialEnvironmentLoop
from mava.types import Action


class OpenSpielSequentialEnvironmentLoop(SequentialEnvironmentLoop):
    """A Sequential MARL environment loop.
    This takes `Environment` and `Executor` instances and coordinates their
    interaction. Executors are updated if `should_update=True`. This can be used as:
        loop = EnvironmentLoop(environment, executor)
        loop.run(num_episodes)
    A `Counter` instance can optionally be given in order to maintain counts
    between different Mava components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.
    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger from acme. A string `label` can be passed
    to easily change the label associated with the default logger; this is ignored
    if a `Logger` instance is given.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        executor: mava.core.Executor,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        should_update: bool = True,
        label: str = "sequential_environment_loop",
    ):
        super().__init__(environment, executor, counter, logger, should_update, label)

    def _get_action(self, agent_id: str, timestep: dm_env.TimeStep) -> List[Action]:
        return [self._executor.select_action(agent_id, timestep.observation)]
