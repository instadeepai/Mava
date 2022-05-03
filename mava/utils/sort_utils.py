import copy
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.random import randint


def atoi(text: str) -> object:
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List:
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def sort_str_num(str_num: Any) -> List[Any]:
    return sorted(str_num, key=natural_keys)


def sample_new_agent_keys(
    agents: List,
    network_sampling_setup: List,
    net_keys_to_ids: Dict[str, int] = None,
    fix_sampler: Optional[List] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Samples new agent networks using the network sampling setup.
    Args:
        agents: List of the agent keys.
        network_sampling_setup: List of networks that are randomly
            sampled from by the executors at the start of an environment run.
            shared_weights: whether agents should share weights or not.
        net_keys_to_ids: Dictionary mapping network keys to network ids.
        fix_sampler: Optional list that can fix the executor sampler to sample
            in a specific way.
    Returns:
        Tuple of dictionaries mapping network keys to ids.
    """
    save_net_keys = {}
    agent_net_keys = {}
    agent_slots = copy.copy(agents)
    loop_i = 0
    while len(agent_slots) > 0:
        if not fix_sampler:
            sample_i = randint(len(network_sampling_setup))
        else:
            sample_i = fix_sampler[loop_i]
        sample = network_sampling_setup[sample_i]
        for net_key in sample:
            agent = agent_slots.pop(0)
            agent_net_keys[agent] = net_key
            if net_keys_to_ids:
                save_net_keys[agent] = np.array(
                    net_keys_to_ids[net_key], dtype=np.int32
                )
        loop_i += 1

    return save_net_keys, agent_net_keys
