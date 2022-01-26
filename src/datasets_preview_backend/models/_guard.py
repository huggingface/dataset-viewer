from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import Status400Error


def guard_blocked_datasets(dataset_name: str) -> None:
    if dataset_name in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")
