import re
from typing import Any, List


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
