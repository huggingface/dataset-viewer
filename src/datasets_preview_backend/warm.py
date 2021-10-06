import os
from tqdm.contrib.concurrent import thread_map  # type: ignore

from datasets_preview_backend.logger import init_logger
from datasets_preview_backend.queries.datasets import get_datasets
from datasets_preview_backend.queries.rows import get_rows
from datasets_preview_backend.utils import get_int_value

DEFAULT_MAX_THREADS = 1


def call_get_rows(dataset: str) -> None:
    try:  # nosec
        get_rows(dataset=dataset)
    except Exception:
        pass


def warm() -> None:
    max_threads = get_int_value(os.environ, "MAX_THREADS", DEFAULT_MAX_THREADS)
    datasets = [d["dataset"] for d in get_datasets()["datasets"]]

    threads = min(max_threads, len(datasets))

    thread_map(call_get_rows, datasets, max_workers=threads, desc="warm the cache (datasets)")


if __name__ == "__main__":
    init_logger(log_level="ERROR")
    warm()
