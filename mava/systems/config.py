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

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Type

from mava.components import Component
from mava.utils.config_utils import flatten_dict


class Config:
    """Config handler for Jax-based Mava systems."""

    def __init__(self) -> None:
        """Initialise config."""
        self._config: Dict = {}
        self._current_params: List = []
        self._built = False

        self._param_to_component: Dict[
            str, str
        ] = {}  # Map from config parameter to which component added it

    def add(self, **kwargs: Any) -> None:
        """Add a component config dataclass.

        Args:
            **kwargs: dictionary with format {name: dataclass}.

        Raises:
            Exception: if a config for an identically named component already exists.
            Exception: if a config shares a parameter name with another config.
            Exception: if a config is not a dataclass object.
        """
        if self._built:
            raise Exception(
                "Component configs cannot be added to an already built config."
            )

        for name, dataclass in kwargs.items():
            if is_dataclass(dataclass):
                if name in list(self._config.keys()):
                    raise Exception(
                        f"The given component ({name}) config is already part of the current \
                        system. Perhaps try updating the component instead using \
                        .update() in the system builder."
                    )
                else:
                    new_param_names = list(dataclass.__dict__.keys())
                    if set(self._current_params) & set(new_param_names):
                        common_parameter_names = set(self._current_params).intersection(
                            set(new_param_names)
                        )
                        raise Exception(
                            f"""
                            Component configs share common parameter names:
                            {common_parameter_names}
                            Components whose configs contain the common parameter names:
                            {[self._param_to_component[common_parameter_name]
                              for common_parameter_name in common_parameter_names]
                             + [name]}.
                            This is not allowed, please ensure config parameter names
                            are unique.
                            """
                        )
                    else:
                        self._current_params.extend(new_param_names)

                        for new_param_name in new_param_names:
                            self._param_to_component[new_param_name] = name
                    self._config[name] = dataclass
            elif isinstance(dataclass, SimpleNamespace):
                # SimpleNamespace implies that
                # this component does not have config variables.
                pass
            else:
                raise Exception(
                    f"""
                    Component configs must be a dataclass.
                    It is type: {type(dataclass)} value: {dataclass}.
                    """
                )

    def update(self, **kwargs: Any) -> None:
        """Update the given component config dataclasses based on their names.

        Args:
            **kwargs: dictionary with format {name: dataclass}.

        Raises:
            Exception: if a config shares a parameter name with another config.
            Exception: if a config is not already part of the system.
            Exception: if a config is not a dataclass object.
        """
        if self._built:
            raise Exception(
                "Component configs cannot be updated if config has already been built."
            )
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
            elif isinstance(dataclass, SimpleNamespace):
                # SimpleNamespace implies that
                # this component does not have config variables.
                pass
            else:
                raise Exception("Component configs must be a dataclass.")

    def build(self) -> None:
        """Build the config file, i.e. unwrap dataclass nested dictionaries.

        Returns:
            None.
        """
        if self._built:
            raise Exception("Config has already been built, this can only happen once.")

        config_unwrapped: Dict = {}
        for param in self._config.values():
            config_unwrapped.update(flatten_dict(param.__dict__))

        self._built_config = config_unwrapped
        self._built = True

    def set_parameters(self, **kwargs: Any) -> None:
        """Set specific hyperparameters of a built config.

        Args:
            **kwargs: dictionary with format {name: parameter value}.

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

        for name, param_value in kwargs.items():
            if name in list(self._built_config.keys()):
                self._built_config[name] = param_value
            else:
                raise Exception(
                    f"""
                    The given parameter ({name}) is not part of the current system.
                    This should have been added first via a component .add() during
                    system building. Ensure that you have defined the correct config
                    class as a type for the component's config init variable. Also
                    ensure that all the component's config variables have types.
                    Current parameters:
                    {list(self._built_config.keys())}.
                    """
                )

    def get(self) -> SimpleNamespace:
        """Get built config for feeding to a Mava system.

        Raises:
            Exception: if trying to get without having first built the config.

        Returns:
            Built config as a SimpleNamespace.
        """
        if not self._built:
            raise Exception(
                "The config must first be built using .build() before calling .get()."
            )

        return SimpleNamespace(**self._built_config)

    def get_local_config(self, component: Type[Component]) -> SimpleNamespace:
        """Get built config for a single component.

        Args:
            component: component to provide config for.

        Returns:
            Built config for a single component.
        """
        if not self._built:
            raise Exception(
                "The config must first be built using .build()"
                "before calling .get_local_config()."
            )

        config_class = component.__init__.__annotations__["config"]

        # Return if there is no config class for the component
        if config_class is SimpleNamespace:
            return config_class()

        # Set local config to global config for names which appear in the config class
        global_config = self._built_config
        local_config: Dict[str, Any] = {}

        for field in fields(config_class):
            local_config[field.name] = global_config[field.name]

        return config_class(**local_config)
