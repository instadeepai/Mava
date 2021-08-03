import copy
import re
from typing import Any, Dict, List, Tuple

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
    executor_samples: List,
    net_to_ints: Dict[str, int] = None,
) -> Tuple[Dict[str, np.array], Dict[str, np.array]]:
    save_net_keys = {}
    agent_net_keys = {}
    agent_slots = copy.copy(agents)
    while len(agent_slots) > 0:
        sample = executor_samples[randint(len(executor_samples))]
        for net_key in sample:
            agent = agent_slots.pop(0)
            agent_net_keys[agent] = net_key
            if net_to_ints:
                save_net_keys[agent] = np.array(net_to_ints[net_key], dtype=np.int32)

    return save_net_keys, agent_net_keys
