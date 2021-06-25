from typing import Any, Dict, Sequence, Union
import time
import numpy as np
import os
from mava.systems.tf import savers as tf2_savers


class VariableSource:
    def __init__(self, variables, checkpoint, checkpoint_subpath) -> None:
        # Init the variable dictionary
        self.variables: Dict[str, Any] = variables

        # Create checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            subdir = os.path.join("variable_source")
            self.checkpointer = tf2_savers.Checkpointer(
                time_delta_minutes=15,
                directory=checkpoint_subpath,
                objects_to_save=self.variables,
                subdirectory=subdir,
            )


            # for agent_key in self.unique_net_keys:
            #     objects_to_save = {
            #         "counter": self._counter,
            #         "policies": self._policy_networks[agent_key],
            #         "critics": self._critic_networks[agent_key],
            #         "observations": self._observation_networks[agent_key],
            #         "target_policies": self._target_policy_networks[agent_key],
            #         "target_critics": self._target_critic_networks[agent_key],
            #         "target_observations": self._target_observation_networks[agent_key],
            #         "policy_optimizer": self._policy_optimizers,
            #         "critic_optimizer": self._critic_optimizers,
            #         "num_steps": self._num_steps,
            #     }

            #     subdir = os.path.join("variable_source", agent_key)
            #     checkpointer = tf2_savers.Checkpointer(
            #         time_delta_minutes=15,
            #         directory=checkpoint_subpath,
            #         objects_to_save=objects_to_save,
            #         subdirectory=subdir,
            #     )
            #     self._system_checkpointer[agent_key] = checkpointer

    def get_variables(
        self, names: Union[str, Sequence[str]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        # import tensorflow as tf
        # tf.print("Getting variable inside source.")
        if type(names) == str:
            return self.variables[names]  # type: ignore
        else:
            variables: Dict[str, Dict[str, np.ndarray]] = {}
            for var_key in names:
                variables[var_key] = self.variables[var_key]
            return variables

    def set_variables(self, names: Sequence[str], vars: Dict[str, np.ndarray]) -> None:
        # import tensorflow as tf
        # tf.print("Setting variable inside source.")
        if type(names) == str:
            vars = {names: vars}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            self.variables[var_key] = vars[var_key]
        return

    def run(self):
        # Checkpoints every 15 minutes
        while True:
            time.sleep(15*60)
            if self.checkpointer:
                self._system_checkpointer.save()
