from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from optax import Params

from mava.components.component import Component
from mava.core_jax import SystemBuilder, SystemParameterServer


@dataclass
class BestCheckpointerConfig:
    checkpointing_metric: Tuple = ("mean_episode_return",)
    checkpoint_best_perf: bool = False
    # Flag to calculate the absolute metric
    absolute_metric: bool = False
    # How many episodes to run evaluation for
    absolute_metric_duration: Optional[int] = 320
    # When to calculate the absolute metric
    absolute_metric_interval: int = 2000000


class BestCheckpointer(Component):
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

        builder.store.best_checkpoint = self.init_checkpointing_params(builder)

    def on_parameter_server_init(self, server: SystemParameterServer) -> None:
        """Adding checkpointing parameters to parameter server"""
        if not self.config.checkpoint_best_perf:
            return

        server.store.parameters.update(
            {"best_checkpoint": self.init_checkpointing_params(server)}
        )

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
