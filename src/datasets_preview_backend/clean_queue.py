import logging

from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import clean_database, connect_to_queue

if __name__ == "__main__":
    init_logger("INFO", "clean_queue")
    logger = logging.getLogger("clean_queue")
    connect_to_queue()
    clean_database()
    logger.info("the queue database is now empty")
