from enum import Enum


class MockedEnvironments(str, Enum):
    Mocked_Dicrete = "discrete_mock"
    Mocked_Continous = "continous_mock"


class EnvType(Enum):
    Sequential = 1
    Parallel = 2


class EnvSpec:
    def __init__(self, env_name: str, env_type: EnvType):
        self.env_name = env_name
        self.env_type = env_type
