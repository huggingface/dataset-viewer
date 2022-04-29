import logging


def init_logger(log_level: str = "INFO", name: str = "datasets_preview_backend") -> None:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(levelname)s: %(asctime)s - %(name)s - %(message)s")

    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    logger.debug(f"Log level set to: {logging.getLevelName(logger.getEffectiveLevel())}")
