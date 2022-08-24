import logging

from libqueue.queue import cancel_started_splits_jobs, connect_to_queue
from libutils.logger import init_logger

from admin.config import LOG_LEVEL, MONGO_QUEUE_DATABASE, MONGO_URL

if __name__ == "__main__":
    init_logger(LOG_LEVEL, "cancel_jobs_splits_next")
    logger = logging.getLogger("cancel_jobs_splits_next")
    connect_to_queue(MONGO_QUEUE_DATABASE, MONGO_URL)
    cancel_started_splits_jobs()
    logger.info("all the started jobs in the splits-next/ queue have been cancelled and re-enqueued")
