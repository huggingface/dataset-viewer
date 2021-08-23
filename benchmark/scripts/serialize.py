from typing import Union

SLASH_SEPARATOR = "___SLASH___"
CONFIG_SEPARATOR = "___CONFIG___"
CONFIG_NONE = "___NONE_CONFIG___"


def serialize_dataset_name(dataset: str) -> str:
    return dataset.replace("/", SLASH_SEPARATOR)


def deserialize_dataset_name(dataset: str) -> str:
    return dataset.replace(SLASH_SEPARATOR, "/")


def serialize_config_name(dataset: str, config: Union[str, None]) -> str:
    c = CONFIG_NONE if config is None else config
    return serialize_dataset_name(dataset) + CONFIG_SEPARATOR + c
