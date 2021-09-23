from typing import Dict, List, Optional, TypedDict

characters: Dict[str, str] = {
    "/": "___SLASH",
    " ": "___SPACE",
    "(": "___PAR",
    ")": "___END_PAR",
}

ordered_keys: Dict[str, str] = {"dataset": "___DATASET", "config": "___CONFIG", "split": "___SPLIT"}


def serialize_params(params: Dict[str, str]) -> str:
    # order is important: "config" can be present only if "dataset" is also present
    s = ""
    for (key, prefix) in ordered_keys.items():
        if key not in params:
            return s
        s += prefix + serialize(params[key])
    return s


def deserialize_params(s: str) -> Dict[str, str]:
    d_all: Dict[str, str] = {}
    # order is important: "config" can be present only if "dataset" is also present
    for (key, prefix) in reversed(ordered_keys.items()):
        s, _, b = s.strip().partition(prefix)
        value = deserialize(b)
        d_all[key] = value

    d: Dict[str, str] = {}
    for (key, prefix) in ordered_keys.items():
        if d_all[key] == "":
            return d
        d[key] = d_all[key]

    return d


def serialize(s: str) -> str:
    for (unsafe, safe) in characters.items():
        s = s.replace(unsafe, safe)
    return s


def deserialize(s: str) -> str:
    for (unsafe, safe) in characters.items():
        s = s.replace(safe, unsafe)
    return s
