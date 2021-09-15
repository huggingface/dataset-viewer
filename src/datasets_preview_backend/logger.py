import logging

from datasets_preview_backend.constants import DEFAULT_LOG_LEVEL


def init_logger(log_level=DEFAULT_LOG_LEVEL):
    logger = logging.getLogger("datasets_preview_backend")
    logger.setLevel(log_level)

    format = "%(levelname)s: %(asctime)s - %(name)s - %(message)s"
    formatter = logging.Formatter(format)

    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    logger.debug(f"Log level set to: {logging.getLevelName(logger.getEffectiveLevel())}")
