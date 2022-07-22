import logging
from typing import List

from dotenv import load_dotenv
from huggingface_hub import list_datasets  # type: ignore
from libcache.cache import (
    connect_to_cache,
    list_split_full_names_to_refresh,
    should_dataset_be_refreshed,
)
from libqueue.queue import add_dataset_job, add_split_job, connect_to_queue
from libutils.logger import init_logger

from admin.config import (
    LOG_LEVEL,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
)

# Load environment variables defined in .env, if any
load_dotenv()


def get_hf_dataset_names():
    return [str(dataset.id) for dataset in list_datasets(full=False)]


def warm_cache(dataset_names: List[str]) -> None:
    logger = logging.getLogger("warm_cache")
    for dataset_name in dataset_names:
        if should_dataset_be_refreshed(dataset_name):
            # don't mark the cache entries as stale, because it's manually triggered
            add_dataset_job(dataset_name)
            logger.info(f"added a job to refresh '{dataset_name}'")
        elif split_full_names := list_split_full_names_to_refresh(dataset_name):
            for split_full_name in split_full_names:
                dataset_name = split_full_name["dataset_name"]
                config_name = split_full_name["config_name"]
                split_name = split_full_name["split_name"]
                # don't mark the cache entries as stale, because it's manually triggered
                add_split_job(dataset_name, config_name, split_name)
                logger.info(
                    f"added a job to refresh split '{split_name}' from dataset '{dataset_name}' with config"
                    f" '{config_name}'"
                )
        else:
            logger.debug(f"dataset already in the cache: '{dataset_name}'")

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
