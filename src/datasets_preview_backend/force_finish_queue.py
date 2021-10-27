import logging

from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import (
    connect_to_queue,
    force_finish_started_jobs,
)

if __name__ == "__main__":
    init_logger("INFO", "force_finish_queue")
    logger = logging.getLogger("force_finish_queue")
    connect_to_queue()
    force_finish_started_jobs()
    logger.info("the started jobs in the queue database have all been marked as finished")
