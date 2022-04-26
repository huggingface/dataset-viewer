import logging

from datasets_preview_backend.io.cache import clean_database, connect_to_cache
from datasets_preview_backend.io.logger import init_logger

if __name__ == "__main__":
    init_logger("INFO", "clean_cache")
    logger = logging.getLogger("clean_cache")
    connect_to_cache()
    clean_database()
    logger.info("the cache database is now empty")
