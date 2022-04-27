import logging

from libutils.logger import init_logger
from libqueue.queue import clean_database, connect_to_queue

if __name__ == "__main__":
    init_logger("INFO", "clean_queues")
    logger = logging.getLogger("clean_queues")
    connect_to_queue()
    clean_database()
    logger.info("the queue database is now empty")
