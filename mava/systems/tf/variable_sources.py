from typing import  Dict, Sequence
from acme.tf import utils as tf2_utils
import numpy as np
import sonnet as snt

class VariableSource():
    def __init__(self) -> None:
        # Init the variable dictionary
        self.variables = {}        

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        if type(names) == str:
            return self.variables[names]
        else:
            variables: Dict[str, Dict[str, np.ndarray]] = {}
            for var_key in names:
                variables[var_key] = self.variables[var_key]
            return variables
    
    def set_variables(self, names: Sequence[str], vars: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        if type(names) == str:
            vars = {names: vars}
            names = [names]

        for var_key in names:
            self.variables[var_key] = vars[var_key]