import logging

from dotenv import load_dotenv

from libcache.cache import (
    connect_to_cache,
    list_split_full_names_to_refresh,
    should_dataset_be_refreshed,
)
from libutils.logger import init_logger
from libqueue.queue import (
    add_dataset_job,
    add_split_job,
    connect_to_queue,
)
from libmodels.hf_dataset import get_hf_dataset_names

# Load environment variables defined in .env, if any
load_dotenv()


def warm() -> None:
    logger = logging.getLogger("warm")
    dataset_names = get_hf_dataset_names()
    for dataset_name in dataset_names:
        split_full_names = list_split_full_names_to_refresh(dataset_name)
        if should_dataset_be_refreshed(dataset_name):
            add_dataset_job(dataset_name)
            logger.info(f"added a job to refresh '{dataset_name}'")
        elif split_full_names:
            for split_full_name in split_full_names:
                dataset_name = split_full_name["dataset_name"]
                config_name = split_full_name["config_name"]
                split_name = split_full_name["split_name"]
                add_split_job(dataset_name, config_name, split_name)
                logger.info(
                    f"added a job to refresh split '{split_name}' from dataset '{dataset_name}' with config"
                    f" '{config_name}'"
                )
        else:
            logger.debug(f"dataset already in the cache: '{dataset_name}'")


if __name__ == "__main__":
    init_logger("INFO", "warm")
    connect_to_cache()
    connect_to_queue()
    warm()
