import logging

from dotenv import load_dotenv

from datasets_preview_backend.io.cache import connect_to_cache
from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import add_dataset_job, connect_to_queue
from datasets_preview_backend.models.hf_dataset import get_hf_dataset_names

# Load environment variables defined in .env, if any
load_dotenv()


def force_refresh_cache() -> None:
    logger = logging.getLogger("force_refresh_cache")
    dataset_names = get_hf_dataset_names()
    for dataset_name in dataset_names:
        add_dataset_job(dataset_name)
    logger.info(f"added {len(dataset_names)} jobs to refresh all the datasets")


if __name__ == "__main__":
    init_logger("INFO", "force_refresh_cache")
    connect_to_cache()
    connect_to_queue()
    force_refresh_cache()
