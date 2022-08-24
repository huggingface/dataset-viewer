import logging
from typing import List

from huggingface_hub.hf_api import HfApi  # type: ignore
from libcache.cache import (
    connect_to_cache,
    list_split_full_names_to_refresh,
    should_dataset_be_refreshed,
)
from libqueue.queue import add_dataset_job, add_split_job, connect_to_queue
from libutils.logger import init_logger

from admin.config import (
    HF_ENDPOINT,
    LOG_LEVEL,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
)


def get_hf_dataset_names():
    return [str(dataset.id) for dataset in HfApi(HF_ENDPOINT).list_datasets(full=False)]


def warm_cache(dataset_names: List[str]) -> None:
    logger = logging.getLogger("warm_cache")
    for dataset in dataset_names:
        if should_dataset_be_refreshed(dataset):
            # don't mark the cache entries as stale, because it's manually triggered
            add_dataset_job(dataset)
            logger.info(f"added a job to refresh '{dataset}'")
        elif split_full_names := list_split_full_names_to_refresh(dataset):
            for split_full_name in split_full_names:
                dataset = split_full_name["dataset"]
                config = split_full_name["config"]
                split = split_full_name["split"]
                # don't mark the cache entries as stale, because it's manually triggered
                add_split_job(dataset, config, split)
                logger.info(f"added a job to refresh split '{split}' from dataset '{dataset}' with config '{config}'")
        else:
            logger.debug(f"dataset already in the cache: '{dataset}'")

    # TODO? also warm splits/ and first-rows/ caches. For now, there are no methods to
    # get access to the stale status, and there is no more logic relation between both cache,
    # so: we should have to read the splits/ cache responses to know which first-rows/ to
    # refresh. It seems a bit too much, and this script is not really used anymore.


if __name__ == "__main__":
    init_logger(LOG_LEVEL, "warm_cache")
    logger = logging.getLogger("warm_cache")
    connect_to_cache(MONGO_CACHE_DATABASE, MONGO_URL)
    connect_to_queue(MONGO_QUEUE_DATABASE, MONGO_URL)
    warm_cache(get_hf_dataset_names())
    logger.info("all the missing datasets have been added to the queue")
