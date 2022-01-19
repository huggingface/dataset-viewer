import logging

from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import (
    connect_to_queue,
    finish_started_dataset_jobs,
    finish_started_split_jobs,
)

if __name__ == "__main__":
    init_logger("INFO", "clean_queues_started")
    logger = logging.getLogger("clean_queues_started")
    connect_to_queue()
    finish_started_dataset_jobs()
    finish_started_split_jobs()
    logger.info("all the started jobs in the queues have been marked as finished")
