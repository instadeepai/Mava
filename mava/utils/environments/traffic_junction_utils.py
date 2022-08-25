import dm_env

from mava.utils.environments.traffic_junction import TrafficJunctionEnv
from mava.wrappers.traffic_junction import TrafficJunctionWrapper


def make_environment(
    difficulty: str = "easy",
    test_for_comms: bool = True,
    evaluation: bool = False,
) -> dm_env.Environment:
    """Create a wrapped Traffic Junction environment.

    Args:
        difficulty: 'easy', 'medium', or 'hard'.
        test_for_comms: Whether to set vision to 0 to test if comms are working.
        evaluation: Whether environment is used by the evaluator.

    Returns:
        Wrapped Traffic Junction environment.
    """
    assert difficulty in {"easy", "medium", "hard"}, "Unsupported difficulty provided."

    del evaluation  # Not used for now

    if difficulty == "easy":
        num_agents = 5
        dim = 6
        add_rate = 0.3
        max_steps = 20
    elif difficulty == "medium":
        num_agents = 10
        dim = 14
        add_rate = 0.2
        max_steps = 40
    else:  # hard
        num_agents = 20
        dim = 18
        add_rate = 0.05
        max_steps = 80

    vision = 0 if test_for_comms else 1  # Set vision to 0 to test comms, 1 is standard

    env_module = TrafficJunctionEnv(
        num_agents=num_agents,
        dim=dim,
        vision=vision,
        add_rate_min=add_rate,
        curr_start=0,
        curr_end=0,
        difficulty=difficulty,
        vocab_type="bool",
        comm_range=3,
    )
    environment = TrafficJunctionWrapper(environment=env_module, max_steps=max_steps)

    return environment
