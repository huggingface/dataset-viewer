import logging

from huggingface_hub import list_datasets  # type: ignore
from libutils.logger import init_logger

from admin.config import LOG_LEVEL
from admin.scripts.refresh_cache import refresh_datasets_cache


def get_hf_canonical_dataset_names():
    return [str(dataset.id) for dataset in list_datasets(full=False) if dataset.id.find("/") == -1]


if __name__ == "__main__":
    init_logger(LOG_LEVEL, "refresh_cache_canonical")
    logger = logging.getLogger("refresh_cache_canonical")
    refresh_datasets_cache(get_hf_canonical_dataset_names())
    logger.info("all the canonical datasets of the Hub have been added to the queue to refresh the cache")
