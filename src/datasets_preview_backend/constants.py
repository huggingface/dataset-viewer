from typing import List, Optional

DEFAULT_APP_HOSTNAME: str = "localhost"
DEFAULT_APP_PORT: int = 8000
DEFAULT_ASSETS_DIRECTORY: None = None
DEFAULT_DATASETS_ENABLE_PRIVATE: bool = False
DEFAULT_DATASETS_REVISION: str = "master"
DEFAULT_EXTRACT_ROWS_LIMIT: int = 100
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_MAX_AGE_LONG_SECONDS: int = 120  # 2 minutes
DEFAULT_MAX_AGE_SHORT_SECONDS: int = 10  # 10 seconds
DEFAULT_MONGO_CACHE_DATABASE: str = "datasets_preview_cache"
DEFAULT_MONGO_QUEUE_DATABASE: str = "datasets_preview_queue"
DEFAULT_MONGO_URL: str = "mongodb://localhost:27018"
DEFAULT_WEB_CONCURRENCY: int = 2

DEFAULT_HF_TOKEN: Optional[str] = None
DEFAULT_MAX_LOAD_PCT: int = 50
DEFAULT_MAX_MEMORY_PCT: int = 60
DEFAULT_MAX_SIZE_FALLBACK: int = 100_000_000
DEFAULT_WORKER_SLEEP_SECONDS: int = 5
DEFAULT_WORKER_QUEUE: str = "datasets"

DEFAULT_REFRESH_PCT: int = 1

# these datasets take too much time, we block them beforehand
DATASETS_BLOCKLIST: List[str] = [
    "imthanhlv/binhvq_news21_raw",
    "SaulLu/Natural_Questions_HTML_Toy",
    "SaulLu/Natural_Questions_HTML_reduced_all",
    "z-uo/squad-it",
    "kiyoung2/aistage-mrc",
    "clips/mqa",
    "Alvenir/nst-da-16khz",
    "fractalego/QA_to_statements",
    "lewtun/gem-multi-dataset-predictions",
    "lukesjordan/worldbank-project-documents",
    "midas/ldke3k_medium",
    "midas/ldke3k_small",
    "midas/ldkp3k_small",
    "qr/cefr_book_sentences",
    "hyperpartisan_news_detection",
    "math_dataset",
    "unicamp-dl/mmarco",
]

FORCE_REDOWNLOAD: str = "force_redownload"
