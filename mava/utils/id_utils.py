from typing import Union

import chex


# should we make a default for type?
@chex.dataclass(frozen=True)
class EntityId:
    id: int
    type: int = 0

    def __lt__(self, other):
        if isinstance(other, str) or isinstance(other, EntityId):
            return str(self) < str(other)
        else:
            raise TypeError(f"Unknown Type used in less than operator: {type(other)}")

    def __str__(self):
        # Using dash to separate to allow underscores in attribute names
        string_rep = ""
        for attribute, value in self.__dict__.items():
            string_rep = f"{string_rep}-{attribute}-{value}"
        return string_rep[1:]

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)

    def index(self, num_agents: int):
        """Returns the index of this agent's items in an observation/reward/etc array"""
        return self.type * num_agents + self.id

    @staticmethod
    def from_string(entity_str: Union[str, "EntityId"]):
        if isinstance(entity_str, str):
            split_string = entity_str.split("-")
            attributes = {}
            for attribute, value in zip(split_string[0::2], split_string[1::2]):
                # TODO is eval slow?
                attributes[attribute] = eval(value)

            return EntityId(**attributes)
        elif isinstance(entity_str, EntityId):
            return entity_str
        else:
            raise TypeError(
                f"Attempted to convert a non-string type: {type(entity_str)}"
            )

    @staticmethod
    def first():
        return EntityId(id=0, type=0)
