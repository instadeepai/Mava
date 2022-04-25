import chex


@chex.dataclass(frozen=True)
class EntityId:
    id: int
    type: int

    def __lt__(self, other):
        return self.id * self.type < other.id * other.type
