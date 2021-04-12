from enum import Enum


class MockedEnvironments(Enum):
    Mocked_Dicrete = 1
    Mocked_Continous = 2


class EnvType(Enum):
    Sequential = 1
    Parallel = 2


class EnvSpec:
    def __init__(self, env_name: str, env_type: EnvType):
        self.env_name = env_name
        self.env_type = env_type
