import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from optax import Params

from mava.components.component import Component
from mava.core_jax import SystemBuilder, SystemParameterServer
from mava.utils.lp_utils import termination_fn


@dataclass
class BestCheckpointerConfig:
    checkpointing_metric: Tuple = ("mean_episode_return",)
    checkpoint_best_perf: bool = False
    # Flag to calculate the absolute metric
    absolute_metric: bool = False
    # How many episodes to run evaluation for
    absolute_metric_duration: Optional[Any] = None


class BestCheckpointer(Component):
    """Best checkpointer is a component that help in storing the best network
    params for two options:
    1- Checkpointing the best network for a specific metric.
    2- Calculating the absolute metric.
    """
    def __init__(
        self, config: BestCheckpointerConfig = BestCheckpointerConfig()
    ) -> None:
        """Initialise BestCheckpointer"""
        super().__init__(config)

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Store checkpointing params in the store"""
        # Save list of metrics attached with their best performance
        builder.store.checkpointing_metric: Dict[str, Any] = {}  # type: ignore
        for metric in list(self.config.checkpointing_metric):
            builder.store.checkpointing_metric[metric] = None

    def on_building_executor_start(self, builder: SystemBuilder) -> None:
        """Initialises the store for best model checkpointing"""
        if not (
            builder.store.is_evaluator
            and (self.config.checkpoint_best_perf or self.config.absolute_metric)
        ):
            return

        if self.config.absolute_metric_duration is None:
            self.config.absolute_metric_duration = (
                10
                * builder.store.global_config.evaluation_duration["evaluator_episodes"]
            )

        builder.store.best_checkpoint = self.init_checkpointing_params(builder)

    def on_parameter_server_init(self, server: SystemParameterServer) -> None:
        """Adding checkpointing parameters to parameter server"""
        if self.config.checkpoint_best_perf:
            # Store the best network in the parameter server just in
            # the case of checkpointing the best performance
            server.store.parameters.update(
                {"best_checkpoint": self.init_checkpointing_params(server)}
            )

        if self.config.absolute_metric:
            if (
                not (server.store.global_config.termination_condition is None)
                and "executor_steps"
                in server.store.global_config.termination_condition.keys()
            ):
                server.calculate_absolute_metric = True  # type: ignore
            else:
                logging.exception(
                    f"{ValueError}: To calculate the absolute metric, you need to define\
                    a termination condition related to the executor_steps."
                )
                termination_fn(server)

    def init_checkpointing_params(
        self, system: Union[SystemParameterServer, SystemBuilder]
    ) -> Dict[str, Any]:
        """Initialises the parameters used for checkpointing the best models"""
        params: Dict[str, Dict[str, Optional[Union[float, Params]]]] = {}
        networks = system.store.networks

        # Create a dictionary of all parameters to save
        for metric in self.config.checkpointing_metric:
            params[metric] = {}
            params[metric]["best_performance"] = None
            for agent_net_key in system.store.networks.keys():
                policy_params = deepcopy(networks[agent_net_key].policy_params)
                params[metric][f"policy_network-{agent_net_key}"] = policy_params

                critic_params = deepcopy(networks[agent_net_key].critic_params)
                params[metric][f"critic_network-{agent_net_key}"] = critic_params

                policy_opt = deepcopy(system.store.policy_opt_states[agent_net_key])
                params[metric][f"policy_opt_state-{agent_net_key}"] = policy_opt

                critic_opt = deepcopy(system.store.critic_opt_states[agent_net_key])
                params[metric][f"critic_opt_state-{agent_net_key}"] = critic_opt

        return params

    @staticmethod
    def name() -> str:
        """Returns the name of the component"""
        return "best_checkpointer"
