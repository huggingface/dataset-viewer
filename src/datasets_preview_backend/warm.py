import os
import time

from datasets_preview_backend.cache_reports import get_kwargs_report
from datasets_preview_backend.logger import init_logger
from datasets_preview_backend.queries.datasets import get_datasets
from datasets_preview_backend.queries.rows import get_rows
from datasets_preview_backend.utils import get_int_value

# Do not hammer the server
DEFAULT_SLEEP_SECONDS = 1


def get_cache_status(dataset: str) -> str:
    report = get_kwargs_report("/rows", {"dataset": dataset})
    return report["status"]


def warm_dataset(dataset: str, sleep_seconds: int) -> None:
    time.sleep(sleep_seconds)
    print(f"Cache warming: dataset '{dataset}'")
    t = time.perf_counter()
    try:  # nosec
        get_rows(dataset=dataset)
    except Exception:
        pass
    elapsed_seconds = time.perf_counter() - t
    print(f"Cache warming: dataset '{dataset}' - done in {elapsed_seconds:.1f}s")


def warm() -> None:
    sleep_seconds = get_int_value(os.environ, "SLEEP_SECONDS", DEFAULT_SLEEP_SECONDS)
    datasets = [d["dataset"] for d in get_datasets()["datasets"]]

    for dataset in datasets:
        status = get_cache_status(dataset)
        if status == "cache_miss":
            warm_dataset(dataset, sleep_seconds)


if __name__ == "__main__":
    init_logger(log_level="ERROR")
    warm()
