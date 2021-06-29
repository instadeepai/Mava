import os
import time
from typing import Any, Dict, Sequence, Union

import numpy as np
from acme.tf import utils as tf2_utils

from mava.systems.tf import savers as tf2_savers


class VariableSource:
    def __init__(
        self, variables: Dict[str, Any], checkpoint: bool, checkpoint_subpath: str
    ) -> None:
        # Init the variable dictionary
        self.variables = variables

        if checkpoint:
            # Only save variables that are not empty.
            save_variables = {}
            for key in self.variables.keys():
                var = self.variables[key]
                # Don't store empty tuple (e.g. empty observation_network) variables
                if not (type(var) == tuple and len(var) == 0):
                    save_variables[key] = variables[key]

            # Create checkpointer
            subdir = os.path.join("variable_source")
            self._system_checkpointer = tf2_savers.Checkpointer(
                time_delta_minutes=15,
                directory=checkpoint_subpath,
                objects_to_save=save_variables,
                subdirectory=subdir,
            )

    def get_variables(
        self, names: Union[str, Sequence[str]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        if type(names) == str:
            return self.variables[names]  # type: ignore
        else:
            variables: Dict[str, Dict[str, np.ndarray]] = {}
            for var_key in names:
                # TODO (dries): Do we really have to convert the variables to
                # numpy each time. Can we not keep the variables in numpy form
                # without the checkpointer complaining?
                variables[var_key] = tf2_utils.to_numpy(self.variables[var_key])
            return variables

    def set_variables(self, names: Sequence[str], vars: Dict[str, np.ndarray]) -> None:
        if type(names) == str:
            vars = {names: vars}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            assert var_key in self.variables
            if type(self.variables[var_key]) == tuple:
                # Loop through tuple
                for var_i in range(len(self.variables[var_key])):
                    self.variables[var_key][var_i].assign(vars[var_key][var_i])
            else:
                self.variables[var_key].assign(vars[var_key])
        return

    def add_to_variables(
        self, names: Sequence[str], vars: Dict[str, np.ndarray]
    ) -> None:
        if type(names) == str:
            vars = {names: vars}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            assert var_key in self.variables
            # Note: Can also use self.variables[var_key] = /
            # self.variables[var_key] + vars[var_key]
            self.variables[var_key].assign_add(vars[var_key])
        return

    def run(self) -> None:
        # Checkpoints every 15 minutes
        while True:
            time.sleep(15 * 60)
            if self._system_checkpointer:
                self._system_checkpointer.save()
                print("Updated variables checkpoint.")
