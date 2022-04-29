import logging

from libutils.logger import init_logger
from libqueue.queue import (
    cancel_started_dataset_jobs,
    cancel_started_split_jobs,
    connect_to_queue,
)

if __name__ == "__main__":
    init_logger("INFO", "cancel_started_jobs")
    logger = logging.getLogger("cancel_started_jobs")
    connect_to_queue()
    cancel_started_dataset_jobs()
    cancel_started_split_jobs()
    logger.info("all the started jobs in the queues have been cancelled")
