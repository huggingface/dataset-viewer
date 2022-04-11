from typing import List, Optional

DEFAULT_APP_HOSTNAME: str = "localhost"
DEFAULT_APP_PORT: int = 8000
DEFAULT_ASSETS_DIRECTORY: None = None
DEFAULT_DATASETS_ENABLE_PRIVATE: bool = False
DEFAULT_DATASETS_REVISION: str = "master"
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_MAX_AGE_LONG_SECONDS: int = 120  # 2 minutes
DEFAULT_MAX_AGE_SHORT_SECONDS: int = 10  # 10 seconds
DEFAULT_MONGO_CACHE_DATABASE: str = "datasets_preview_cache"
DEFAULT_MONGO_QUEUE_DATABASE: str = "datasets_preview_queue"
DEFAULT_MONGO_URL: str = "mongodb://localhost:27018"
DEFAULT_WEB_CONCURRENCY: int = 2

DEFAULT_HF_TOKEN: Optional[str] = None
DEFAULT_MAX_JOBS_PER_DATASET: int = 2
DEFAULT_MAX_LOAD_PCT: int = 50
DEFAULT_MAX_MEMORY_PCT: int = 60
DEFAULT_MAX_SIZE_FALLBACK: int = 100_000_000
DEFAULT_ROWS_MAX_BYTES: int = 1_000_000
DEFAULT_ROWS_MAX_NUMBER: int = 100
DEFAULT_ROWS_MIN_NUMBER: int = 10
DEFAULT_WORKER_SLEEP_SECONDS: int = 5
DEFAULT_WORKER_QUEUE: str = "datasets"

DEFAULT_REFRESH_PCT: int = 1

# below 100 bytes, the cell content will not be truncated
DEFAULT_MIN_CELL_BYTES: int = 100

# these datasets take too much time, we block them beforehand
DATASETS_BLOCKLIST: List[str] = [
    "Alvenir/nst-da-16khz",
    "bigscience/P3",
    "clips/mqa",
    "echarlaix/gqa-lxmert",
    "echarlaix/vqa-lxmert",
    "fractalego/QA_to_statements",
    "hyperpartisan_news_detection",
    "imthanhlv/binhvq_news21_raw",
    "Graphcore/gqa-lxmert",
    "Graphcore/vqa-lxmert",
    "kiyoung2/aistage-mrc",
    "lewtun/gem-multi-dataset-predictions",
    "lukesjordan/worldbank-project-documents",
    "math_dataset",
    "midas/ldke3k_medium",
    "midas/ldke3k_small",
    "midas/ldkp3k_small",
    "qr/cefr_book_sentences",
    "SaulLu/Natural_Questions_HTML_reduced_all",
    "SaulLu/Natural_Questions_HTML_Toy",
    "unicamp-dl/mmarco",
    "z-uo/squad-it",
]
