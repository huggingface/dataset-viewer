from typing import Tuple

SLASH = "___SLASH___"
SPACE = "___SPACE___"
PAR_OPEN = "___PAR_OPEN___"
PAR_CLOSE = "___PAR_CLOSE___"
CONFIG_SEPARATOR = "___CONFIG___"
SPLIT_SEPARATOR = "___SPLIT___"


def serialize_dataset_name(dataset: str) -> str:
    return dataset.replace("/", SLASH)


def deserialize_dataset_name(serialized_dataset: str) -> str:
    return serialized_dataset.replace(SLASH, "/")


def serialize_config_name(dataset: str, config: str) -> str:
    # due to config named "(China)", "bbc hindi nli"
    safe_config = config.replace("(", PAR_OPEN).replace(")", PAR_CLOSE).replace(" ", SPACE)
    return serialize_dataset_name(dataset) + CONFIG_SEPARATOR + safe_config


def deserialize_config_name(serialized_config: str) -> Tuple[str, str]:
    serialized_dataset, _, safe_config = serialized_config.partition(CONFIG_SEPARATOR)
    config = safe_config.replace(PAR_OPEN, "(").replace(PAR_CLOSE, ")").replace(SPACE, " ")
    dataset = deserialize_dataset_name(serialized_dataset)
    return dataset, config


def serialize_split_name(dataset: str, config: str, split: str) -> str:
    safe_split = split
    return serialize_config_name(dataset, config) + SPLIT_SEPARATOR + safe_split


def deserialize_split_name(serialized_split: str) -> Tuple[str, str, str]:
    serialized_config, _, safe_split = serialized_split.partition(SPLIT_SEPARATOR)
    split = safe_split.replace(PAR_OPEN, "(").replace(PAR_CLOSE, ")")
    dataset, config = deserialize_config_name(serialized_config)
    return dataset, config, split
