from open_spiel.python import rl_environment  # type: ignore


def load_open_spiel_env(game_name: str) -> rl_environment.Environment:
    """Loads an open spiel environment given a game name Also, the possible agents in the
    environment are set"""

    env = rl_environment.Environment(game_name)
    env.agents = [f"player_{i}" for i in range(env.num_players)]
    env.possible_agents = env.agents[:]

    return env
