from typing import List, Optional, TypedDict

from datasets import get_dataset_config_names

from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME, FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.models.info import Info, get_info
from datasets_preview_backend.models.split import Split, get_splits


class Config(TypedDict):
    config_name: str
    splits: List[Split]
    info: Info


def get_config(dataset_name: str, config_name: str, hf_token: Optional[str] = None) -> Config:
    if not isinstance(config_name, str):
        raise TypeError("config_name argument should be a string")
    # Get all the data
    info = get_info(dataset_name, config_name, hf_token)
    splits = get_splits(dataset_name, config_name, info, hf_token)

    return {"config_name": config_name, "splits": splits, "info": info}


def get_config_names(dataset_name: str, hf_token: Optional[str] = None) -> List[str]:
    try:
        config_names: List[str] = get_dataset_config_names(
            dataset_name, download_mode=FORCE_REDOWNLOAD, use_auth_token=hf_token  # type: ignore
        )
        if not config_names:
            config_names = [DEFAULT_CONFIG_NAME]
    except Exception as err:
        raise Status400Error("Cannot get the config names for the dataset.", err)
    return config_names


def get_configs(dataset_name: str, hf_token: Optional[str] = None) -> List[Config]:
    return [
        get_config(dataset_name, config_name, hf_token) for config_name in get_config_names(dataset_name, hf_token)
    ]
