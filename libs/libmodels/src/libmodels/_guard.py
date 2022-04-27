from libutils.exceptions import Status400Error

from libmodels.constants import DATASETS_BLOCKLIST


def guard_blocked_datasets(dataset_name: str) -> None:
    if dataset_name in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")
