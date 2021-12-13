import logging

from dotenv import load_dotenv

from datasets_preview_backend.io.cache import connect_to_cache, is_dataset_cached
from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import add_job, connect_to_queue
from datasets_preview_backend.models.hf_dataset import get_hf_dataset_names

# Load environment variables defined in .env, if any
load_dotenv()


def warm() -> None:
    logger = logging.getLogger("warm")
    dataset_names = get_hf_dataset_names()
    for dataset_name in dataset_names:
        if not is_dataset_cached(dataset_name):
            add_job(dataset_name)
            logger.info(f"added a job to refresh '{dataset_name}'")
        else:
            logger.debug(f"dataset already in the cache: '{dataset_name}'")


if __name__ == "__main__":
    init_logger("INFO", "warm")
    connect_to_cache()
    connect_to_queue()
    warm()
