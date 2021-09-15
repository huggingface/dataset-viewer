import logging

from datasets import get_dataset_config_names

from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error


def get_configs(dataset: str):
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    try:
        configs = get_dataset_config_names(dataset)
        if len(configs) == 0:
            configs = [DEFAULT_CONFIG_NAME]
    except FileNotFoundError as err:
        raise Status404Error("The dataset could not be found.") from err
    except Exception as err:
        raise Status400Error("The config names could not be parsed from the dataset.") from err

    return {"dataset": dataset, "configs": configs}
