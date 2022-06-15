import logging
from typing import List

from dotenv import load_dotenv
from huggingface_hub import list_datasets  # type: ignore
from libqueue.queue import add_dataset_job, connect_to_queue
from libutils.logger import init_logger

from admin.config import LOG_LEVEL, MONGO_QUEUE_DATABASE, MONGO_URL

# Load environment variables defined in .env, if any
load_dotenv()


def get_hf_dataset_names():
    return [str(dataset.id) for dataset in list_datasets(full=True)]


def refresh_datasets_cache(dataset_names: List[str]) -> None:
    logger = logging.getLogger("warm_cache")
    for dataset_name in dataset_names:
        add_dataset_job(dataset_name)
        logger.info(f"added a job to refresh '{dataset_name}'")


if __name__ == "__main__":
    init_logger(LOG_LEVEL, "warm_cache")
    logger = logging.getLogger("warm_cache")
    connect_to_queue(MONGO_QUEUE_DATABASE, MONGO_URL)
    refresh_datasets_cache(get_hf_dataset_names())
    logger.info("all the datasets of the Hub have been added to the queue to refresh the cache")
