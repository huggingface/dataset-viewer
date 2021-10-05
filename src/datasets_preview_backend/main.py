from datasets_preview_backend.app import start
from datasets_preview_backend.cache import show_cache_dir  # type: ignore
from datasets_preview_backend.config import LOG_LEVEL
from datasets_preview_backend.logger import init_logger

if __name__ == "__main__":
    init_logger(log_level=LOG_LEVEL)
    show_cache_dir()
    start()
