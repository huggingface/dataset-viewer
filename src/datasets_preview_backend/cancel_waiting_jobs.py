import logging

from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import (
    connect_to_queue,
    finish_waiting_dataset_jobs,
    finish_waiting_split_jobs,
)

if __name__ == "__main__":
    init_logger("INFO", "clean_queues_waiting")
    logger = logging.getLogger("clean_queues_waiting")
    connect_to_queue()
    finish_waiting_dataset_jobs()
    finish_waiting_split_jobs()
    logger.info("all the waiting jobs in the queues have been marked as finished")
