from typing import Any, Dict, Sequence, Union

import numpy as np


class VariableSource:
    def __init__(self) -> None:
        # Init the variable dictionary
        self.variables: Dict[str, Any] = {}

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
