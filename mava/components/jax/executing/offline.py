from mava.components.jax import Component
from chex import dataclass
from mava.wrappers.offline_environment_logger import MAOfflineEnvironmentSequenceLogger
from mava.core_jax import SystemExecutor


@dataclass
class EvaluatorOfflineLoggingConfig:
    sequence_length: int = 1000
    period: int = 100
    logdir: str = "~./offline_env_logs"
    label: str = "offline_logger"
    min_sequences_per_file: int = 1000


class EvaluatorOfflineLogging(Component):
    def __init__(
        self,
        config: EvaluatorOfflineLoggingConfig = EvaluatorOfflineLoggingConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_execution_init_end(self, executor: SystemExecutor):
        if executor.store.is_evaluator:
            executor.store.environment_loop._environment = (
                MAOfflineEnvironmentSequenceLogger(
                    executor.store.environment_loop._environment,
                    self.config.sequence_length,
                    self.config.period,
                    self.config.logdir,
                    self.config.label,
                    self.config.min_sequences_per_file,
                )
            )

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "evaluator_offline_logging"  # for creating system lowercase underscore

    @staticmethod
    def config_class():
        return EvaluatorOfflineLoggingConfig
