# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config class for Mava systems"""

from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import Any, Dict, List

from mava.utils.config_utils import flatten_dict


class Config:
    """Config handler for Jax-based Mava systems."""

    def __init__(self) -> None:
        """Initialise config"""
        self._config: Dict = {}
        self._current_params: List = []
        self._built = False

    def add(self, **kwargs: Any) -> None:
        """Add a component config dataclass.

        Raises:
            Exception: if a config for an identically named component already exists
            Exception: if a config shares a parameter name with another config
            Exception: if a config is not a dataclass object
        """
        if not self._built:
            for name, dataclass in kwargs.items():
                if is_dataclass(dataclass):
                    if name in list(self._config.keys()):
                        raise Exception(
                            "The given component config is already part of the current \
                            system. Perhaps try updating the component instead using \
                            .update() in the system builder."
                        )
                    else:
                        new_param_names = list(dataclass.__dict__.keys())
                        if set(self._current_params) & set(new_param_names):
                            raise Exception(
                                "Component configs share a common parameter name. \
                                This is not allowed, please ensure config \
                                parameter names are unique."
                            )
                        else:
                            self._current_params.extend(new_param_names)
                        self._config[name] = dataclass
                else:
                    raise Exception("Component configs must be a dataclass.")
        else:
            raise Exception(
                "Component configs cannot be added to an already built config."
            )

    def update(self, **kwargs: Any) -> None:
        """Update a component config dataclass.

        Raises:
            Exception: if a config shares a parameter name with another config
            Exception: if a config is not already part of the system
            Exception: if a config is not a dataclass object
        """
        if not self._built:
            for name, dataclass in kwargs.items():
                if is_dataclass(dataclass):
                    if name in list(self._config.keys()):
                        # When updating a component, the list of current parameter names
                        # might contain the parameter names of the new component
                        # with additional new parameter names that still need to be
                        # checked with other components. Therefore, we first take the
                        # difference between the current set and the component being
                        # updated.
                        self._current_params = list(
                            set(self._current_params).difference(
                                list(self._config[name].__dict__.keys())
                            )
                        )
                        new_param_names = list(dataclass.__dict__.keys())
                        if set(self._current_params) & set(new_param_names):
                            raise Exception(
                                "Component configs share a common parameter name. \
                                This is not allowed, please ensure config \
                                parameter names are unique."
                            )
                        else:
                            self._current_params.extend(new_param_names)
                            self._config[name] = dataclass
                    else:
                        raise Exception(
                            "The given component config is not part of the current \
                            system. Perhaps try adding the component using .add() \
                            in the system builder."
                        )
                else:
                    raise Exception("Component configs must be a dataclass.")
        else:
            raise Exception(
                "Component configs cannot be updated if config has already been built."
            )

    def build(self) -> None:
        """Build the config file, i.e. unwrap dataclass nested dictionaries"""
        if not self._built:
            config_unwrapped: Dict = {}
            for param in self._config.values():
                config_unwrapped.update(flatten_dict(param.__dict__))
            self._built_config = config_unwrapped
            self._built = True
        else:
            raise Exception("Config has already been built, this can only happen once.")

    def set_parameters(self, **kwargs: Any) -> None:
        """Set a specific hyperparameter of a built config.

        Raises:
            Exception: if a set is attempted on a config not yet built.
            Exception: if a set is attempted for a hyperparameter that is not part \
                of the built config.
        """

        if not self._built:
            raise Exception(
                "Config must first be built using .build() before hyperparameters \
                can be set to different values using .set()."
            )
        else:
            for name, param_value in kwargs.items():
                if name in list(self._built_config.keys()):
                    self._built_config[name] = param_value
                else:
                    raise Exception(
                        "The given parameter is not part of the current system. \
                        This should have been added first via a component .add() \
                        during system building."
                    )

    def get(self) -> SimpleNamespace:
        """Get built config for feeding to a Mava system.

        Raises:
            Exception: if trying to get without having first built the config
        Returns:
            built config
        """
        if self._built:
            return SimpleNamespace(**self._built_config)
        else:
            raise Exception(
                "The config must first be built using .build() before calling .get()."
            )
