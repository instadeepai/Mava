from mava.utils.debugging.core import World


# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self, num_agents: int) -> World:
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world: World) -> None:
        raise NotImplementedError()
