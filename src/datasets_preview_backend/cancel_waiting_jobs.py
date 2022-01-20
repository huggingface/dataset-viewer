import logging

from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import (
    cancel_waiting_dataset_jobs,
    cancel_waiting_split_jobs,
    connect_to_queue,
)

if __name__ == "__main__":
    init_logger("INFO", "cancel_waiting_jobs")
    logger = logging.getLogger("cancel_waiting_jobs")
    connect_to_queue()
    cancel_waiting_dataset_jobs()
    cancel_waiting_split_jobs()
    logger.info("all the waiting jobs in the queues have been cancelled")
