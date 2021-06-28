import os
import time
from typing import Any, Dict, Sequence, Union

import numpy as np
from acme.tf import utils as tf2_utils
from mava.systems.tf import savers as tf2_savers


class VariableSource:
    def __init__(self, variables, checkpoint, checkpoint_subpath) -> None:
        # Init the variable dictionary
        self.variables: Dict[str, Any] = variables

        if checkpoint:
            # Only save variables that are not empty.
            save_variables = {}
            for key in self.variables.keys():
                var = self.variables[key]
                # Don't store empty tuple (e.g. empty observation_network) variables
                if not (type(var)==tuple and len(var) == 0):
                    save_variables[key] = variables[key]

            # Create checkpointer
            subdir = os.path.join("variable_source")
            self.checkpointer = tf2_savers.Checkpointer(
                time_delta_minutes=15,
                directory=checkpoint_subpath,
                objects_to_save=save_variables,
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
        if type(names) == str:
            return self.variables[names]  # type: ignore
        else:
            variables: Dict[str, Dict[str, np.ndarray]] = {}
            for var_key in names:
                # TODO (dries): Do we really have to convert the variables to numpy each time. Can we not keep
                # the variables in numpy form without the checkpointer complaining?
                variables[var_key] = tf2_utils.to_numpy(self.variables[var_key])
                # 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
            return variables

    def set_variables(self, names: Sequence[str], vars: Dict[str, np.ndarray]) -> None:
        # import tensorflow as tf
        # tf.print("Setting variable inside source.")
        if type(names) == str:
            vars = {names: vars}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            assert var_key in self.variables
            self.variables[var_key] = vars[var_key]
        return

    def add_to_variables(self, names: Sequence[str], vars: Dict[str, np.ndarray]) -> None:
        if type(names) == str:
            vars = {names: vars}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            assert var_key in self.variables
            self.variables[var_key] += vars[var_key]
        return

    def run(self):
        # Checkpoints every 15 minutes
        while True:
            time.sleep(15 * 60)
            if self.checkpointer:
                self._system_checkpointer.save()
                print("Updated variables checkpoint.")
