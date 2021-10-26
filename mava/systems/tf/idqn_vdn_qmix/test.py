import numpy as np
from flatland.envs.step_utils.states import TrainState

from utils import create_rail_env

ENV_PARAMS = {
        # Test_0
        "n_agents": 1,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "malfunction_rate": 1 / 200,
        "seed": 0,
        "observation_tree_depth": 2,
        "observation_max_path_depth": 30
}

def act(obs):
    actions = {}
    for agent in obs.keys():
        actions[agent] = np.random.choice(5)

    return actions

env = create_rail_env(**ENV_PARAMS)

while True:
    timestep, info = env.reset()

    episode_return = 0
    episode_steps = 0

    # Run an episode.
    while not timestep.last():
        actions = act(timestep.observation)

        timestep, info = env.step(actions)

        env.render()

        # Book-keeping.
        episode_steps += 1

    episode_return += sum([r for r in timestep.reward.values()])

    tasks_finished = sum([1 if info['state'][agent] == TrainState.DONE else 0 for agent in env.agents])
    completion = tasks_finished / env.get_num_agents()
    normalized_score = episode_return / (env._max_episode_steps * env.get_num_agents())

    result = {
            "episode_length": episode_steps,
            "episode_return": episode_return,
            "score": normalized_score,
            "completion": completion,
        }

    print(result)