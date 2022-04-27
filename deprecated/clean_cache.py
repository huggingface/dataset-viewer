import logging

from libcache.cache import clean_database, connect_to_cache
from libutils.logger import init_logger

if __name__ == "__main__":
    init_logger("INFO", "clean_cache")
    logger = logging.getLogger("clean_cache")
    connect_to_cache()
    clean_database()
    logger.info("the cache database is now empty")
