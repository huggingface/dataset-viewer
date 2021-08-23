from typing import Tuple, Union

SLASH = "___SLASH___"
SPACE = "___SPACE___"
PAR_OPEN = "___PAR_OPEN___"
PAR_CLOSE = "___PAR_CLOSE___"
CONFIG_SEPARATOR = "___CONFIG___"
CONFIG_NONE = "___NONE_CONFIG___"
SPLIT_SEPARATOR = "___SPLIT___"


def serialize_dataset_name(dataset: str) -> str:
    return dataset.replace("/", SLASH)


def deserialize_dataset_name(serialized_dataset: str) -> str:
    return serialized_dataset.replace(SLASH, "/")


def serialize_config_name(dataset: str, config: Union[str, None]) -> str:
    c = CONFIG_NONE if config is None else config
    # due to config named "(China)", "bbc hindi nli"
    safe_config = c.replace("(", PAR_OPEN).replace(")", PAR_CLOSE).replace(" ", SPACE)
    return serialize_dataset_name(dataset) + CONFIG_SEPARATOR + safe_config


def deserialize_config_name(serialized_config: str) -> Tuple[str, Union[str, None]]:
    serialized_dataset, _, safe_config = serialized_config.partition(CONFIG_SEPARATOR)
    c = safe_config.replace(PAR_OPEN, "(").replace(PAR_CLOSE, ")").replace(SPACE, " ")
    config = None if c == CONFIG_NONE else c
    dataset = deserialize_dataset_name(serialized_dataset)
    return dataset, config


def serialize_split_name(dataset: str, config: Union[str, None], split: str) -> str:
    safe_split = split
    return serialize_config_name(dataset, config) + SPLIT_SEPARATOR + safe_split


def deserialize_split_name(serialized_split: str) -> Tuple[str, Union[str, None], str]:
    serialized_config, _, safe_split = serialized_split.partition(SPLIT_SEPARATOR)
    split = safe_split.replace(PAR_OPEN, "(").replace(PAR_CLOSE, ")")
    dataset, config = deserialize_config_name(serialized_config)
    return dataset, config, split
