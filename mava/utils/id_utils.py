import chex


@chex.dataclass(frozen=True)
class EntityId:
    type: int
    id: int

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
        return string_rep

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)

    @staticmethod
    def from_string(other: str):
        split_string = other.split("-")
        attributes = {}
        for attribute, value in zip(split_string[1::2], split_string[2::2]):
            attributes[attribute] = value

        return EntityId(**attributes)
