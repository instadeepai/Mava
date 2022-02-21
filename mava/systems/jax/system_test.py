# # python3
# # Copyright 2021 InstaDeep Ltd. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Tests for Jax-based Mava system implementation."""
# from dataclasses import dataclass
# from typing import Any, List

# import pytest

# from mava.systems.jax import System


# # test components
# @dataclass
# class ComponentZeroDefaultConfig:
#     param_0: int
#     param_1: str


# class ComponentZero:
#     def __init__(self, config=None) -> None:
#         self.config = config if config else ComponentZeroDefaultConfig()

#     def dummy_int_plus_str(self) -> int:
#         return self.config.param_0 + int(self.config.param_1)

#     @property
#     def name(self):
#         return self.__class__.__name__


# @dataclass
# class ComponentOneDefaultConfig:
#     param_2: float
#     param_3: bool


# class ComponentOne:
#     def __init__(self, config=None) -> None:
#         self.config = config if config else ComponentOneDefaultConfig()

#     def dummy_float_plus_bool(self) -> float:
#         return self.config.param_2 + float(self.config.param_3)

#     @property
#     def name(self):
#         return self.__class__.__name__


# @dataclass
# class ComponentTwoDefaultConfig:
#     param_4: str
#     param_5: bool


# class ComponentTwo:
#     def __init__(self, config=None) -> None:
#         self.config = config if config else ComponentTwoDefaultConfig()

#     def dummy_str_plus_bool(self) -> int:
#         return int(self.config.param_4) + int(self.config.param_5)

#     @property
#     def name(self):
#         return self.__class__.__name__


# class TestSystem(System):
#     def design(self) -> List[Any]:
#         components = [ComponentZero, ComponentOne]
#         return components


# @pytest.fixture
# def system() -> None:
#     return TestSystem()


# def test_default_config(system) -> None:
#     pass


# def test_system_update(system) -> None:
#     pass


# def test_system_add(system) -> None:
#     pass


# def test_system_launch(system) -> None:
#     pass
