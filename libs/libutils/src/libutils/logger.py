import logging

from libutils.constants import DEFAULT_LOG_LEVEL


def init_logger(log_level: str = DEFAULT_LOG_LEVEL, name: str = "datasets_preview_backend") -> None:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    format = "%(levelname)s: %(asctime)s - %(name)s - %(message)s"
    formatter = logging.Formatter(format)

    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    # set logs from the datasets library to the least verbose
    # datasets.utils.logging.set_verbosity(datasets.utils.logging.log_levels["critical"])
    # TODO: move somewhere else, or remove

    logger.debug(f"Log level set to: {logging.getLevelName(logger.getEffectiveLevel())}")
