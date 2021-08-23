from typing import Tuple, Union

SLASH_SEPARATOR = "___SLASH___"
CONFIG_SEPARATOR = "___CONFIG___"
CONFIG_NONE = "___NONE_CONFIG___"
SPLIT_SEPARATOR = "___SPLIT___"


def serialize_dataset_name(dataset: str) -> str:
    return dataset.replace("/", SLASH_SEPARATOR)


def deserialize_dataset_name(serialized_dataset: str) -> str:
    return serialized_dataset.replace(SLASH_SEPARATOR, "/")


def serialize_config_name(dataset: str, config: Union[str, None]) -> str:
    c = CONFIG_NONE if config is None else config
    return serialize_dataset_name(dataset) + CONFIG_SEPARATOR + c


def deserialize_config_name(serialized_config: str) -> Tuple[str, Union[str, None]]:
    serialized_dataset, _, safe_config = serialized_config.partition(CONFIG_SEPARATOR)
    config = None if safe_config == CONFIG_NONE else safe_config
    dataset = deserialize_dataset_name(serialized_dataset)
    return dataset, config


def serialize_split_name(dataset: str, config: Union[str, None], split: str) -> str:
    return serialize_config_name(dataset, config) + SPLIT_SEPARATOR + split


def deserialize_split_name(serialized_split: str) -> Tuple[str, Union[str, None], str]:
    serialized_config, _, split = serialized_split.partition(SPLIT_SEPARATOR)
    dataset, config = deserialize_config_name(serialized_config)
    return dataset, config, split
