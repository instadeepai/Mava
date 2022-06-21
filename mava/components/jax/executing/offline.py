from mava.components.jax import Component
from mava.components.jax.building.environments import (
    ExecutorEnvironmentLoopConfig,
    ParallelExecutorEnvironmentLoop,
)
from mava.core_jax import SystemBuilder, SystemExecutor
from mava.wrappers.offline_environment_logger import MAOfflineEnvironmentSequenceLogger
from chex import dataclass


@dataclass
class EvaluatorOfflineLoggingConfig(ExecutorEnvironmentLoopConfig):
    offline_sequence_length: int = 1000
    offline_sequence_period: int = 1000
    offline_logdir: str = "./offline_env_logs"
    offline_label: str = "offline_logger"
    offline_min_sequences_per_file: int = 100


class EvaluatorOfflineLogging(ParallelExecutorEnvironmentLoop):
    def __init__(
        self,
        config: EvaluatorOfflineLoggingConfig = EvaluatorOfflineLoggingConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_executor_environment(self, builder: SystemBuilder):
        env = self.config.environment_factory(evaluation=False)  # type: ignore

        if builder.store.is_evaluator:
            env = MAOfflineEnvironmentSequenceLogger(
                environment=env,
                sequence_length=self.config.offline_sequence_length,
                period=self.config.offline_sequence_period,
                logdir=self.config.offline_logdir,
                label=self.config.offline_label,
                min_sequences_per_file=self.config.offline_min_sequences_per_file,
            )

        builder.store.executor_environment = env

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "executor_environment_loop"  # "evaluator_offline_logging"  # for creating system lowercase underscore

    @staticmethod
    def config_class():
        return EvaluatorOfflineLoggingConfig
