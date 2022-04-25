import chex


@chex.dataclass(frozen=True)
class EntityId:
    id: int
    type: int

    def __lt__(self, other):
        return self.id * (self.type + 1) < other.id * (other.type + 1)

    def __str__(self):
        return f"{self.type}_{self.id}"

    def __bytes__(self):
        return b"" + str(self.type) + "_" + str(self.id)
