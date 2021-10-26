import logging
from typing import List, TypedDict

from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.models.config import Config, get_configs

logger = logging.getLogger(__name__)


class Dataset(TypedDict):
    dataset_name: str
    configs: List[Config]


def get_dataset(dataset_name: str) -> Dataset:
    if dataset_name in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")
    configs = get_configs(dataset_name)
    return {"dataset_name": dataset_name, "configs": configs}
