import os
import time

import psutil  # type: ignore
from dotenv import load_dotenv

from datasets_preview_backend.constants import (
    DEFAULT_MAX_LOAD_PCT,
    DEFAULT_MAX_SWAP_MEMORY_PCT,
    DEFAULT_MAX_VIRTUAL_MEMORY_PCT,
)
from datasets_preview_backend.dataset_entries import (
    get_dataset_cache_status,
    get_dataset_entry,
    get_refreshed_dataset_names,
)
from datasets_preview_backend.logger import init_logger
from datasets_preview_backend.utils import get_int_value

# Load environment variables defined in .env, if any
load_dotenv()


def wait_until_load_is_ok(max_load_pct: int) -> None:
    t = time.perf_counter()
    while True:
        load_pct = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        if load_pct[0] < max_load_pct:
            break
        elapsed_seconds = time.perf_counter() - t
        print(f"Waiting ({elapsed_seconds:.1f}s) for the load to be under {max_load_pct}%")
        time.sleep(1)


def warm_dataset(dataset: str, max_load_pct: int) -> None:
    wait_until_load_is_ok(max_load_pct)

    print(f"Cache warming: dataset '{dataset}'")
    t = time.perf_counter()
    try:  # nosec
        get_dataset_entry(dataset=dataset)
    except Exception:
        pass
    elapsed_seconds = time.perf_counter() - t
    print(f"Cache warming: dataset '{dataset}' - done in {elapsed_seconds:.1f}s")


def warm() -> None:
    max_load_pct = get_int_value(os.environ, "MAX_LOAD_PCT", DEFAULT_MAX_LOAD_PCT)
    max_virtual_memory_pct = get_int_value(os.environ, "MAX_VIRTUAL_MEMORY_PCT", DEFAULT_MAX_VIRTUAL_MEMORY_PCT)
    max_swap_memory_pct = get_int_value(os.environ, "MAX_SWAP_MEMORY_PCT", DEFAULT_MAX_SWAP_MEMORY_PCT)
    datasets = get_refreshed_dataset_names()

    for dataset in datasets:
        if psutil.virtual_memory().percent > max_virtual_memory_pct:
            print("Memory usage is too high, we stop here.")
            return
        if psutil.swap_memory().percent > max_swap_memory_pct:
            print("Swap memory usage is too high, we stop here.")
            return

        print(f"Checking dataset '{dataset}'")
        status = get_dataset_cache_status(dataset)["status"]
        print(f"Checked: '{status}'")
        if status == "cache_miss":
            warm_dataset(dataset, max_load_pct)


if __name__ == "__main__":
    init_logger(log_level="INFO")
    warm()
