import os
import time
from random import random
from typing import Any

import psutil  # type: ignore
from dotenv import load_dotenv

from datasets_preview_backend.constants import (
    DEFAULT_MAX_LOAD_PCT,
    DEFAULT_MAX_SWAP_MEMORY_PCT,
    DEFAULT_MAX_VIRTUAL_MEMORY_PCT,
    DEFAULT_REFRESH_PCT,
)
from datasets_preview_backend.io.cache import update_dataset_cache
from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.models.hf_dataset import get_hf_dataset_names
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
        print(f"Waiting ({elapsed_seconds:.1f}s) for the load to be under {max_load_pct}%", flush=True)
        time.sleep(1)


def refresh_dataset(dataset_name: str, max_load_pct: int) -> None:
    wait_until_load_is_ok(max_load_pct)

    print(f"Cache refreshing: dataset '{dataset_name}'", flush=True)
    t = time.perf_counter()
    try:  # nosec
        update_dataset_cache(dataset_name)
    except Exception:
        pass
    elapsed_seconds = time.perf_counter() - t
    print(f"Cache refreshing: dataset '{dataset_name}' - done in {elapsed_seconds:.1f}s", flush=True)


# TODO: we could get the first N, sorted by creation time (more or less expire time)
def make_criterion(threshold: float) -> Any:
    return lambda x: random() < threshold  # nosec


def refresh() -> None:
    max_load_pct = get_int_value(os.environ, "MAX_LOAD_PCT", DEFAULT_MAX_LOAD_PCT)
    max_virtual_memory_pct = get_int_value(os.environ, "MAX_VIRTUAL_MEMORY_PCT", DEFAULT_MAX_VIRTUAL_MEMORY_PCT)
    max_swap_memory_pct = get_int_value(os.environ, "MAX_SWAP_MEMORY_PCT", DEFAULT_MAX_SWAP_MEMORY_PCT)
    refresh_pct = get_int_value(os.environ, "REFRESH_PCT", DEFAULT_REFRESH_PCT)

    # TODO: cache get_hf_dataset_names?
    dataset_names = get_hf_dataset_names()

    criterion = make_criterion(refresh_pct / 100)
    selected_dataset_names = list(filter(criterion, dataset_names))
    print(
        f"Refreshing: {len(selected_dataset_names)} datasets from"
        f" {len(dataset_names)} ({100*len(selected_dataset_names)/len(dataset_names):.1f}%)",
        flush=True,
    )

    for dataset_name in selected_dataset_names:
        if psutil.virtual_memory().percent > max_virtual_memory_pct:
            print("Memory usage is too high, we stop here.", flush=True)
            return
        if psutil.swap_memory().percent > max_swap_memory_pct:
            print("Swap memory usage is too high, we stop here.", flush=True)
            return
        refresh_dataset(dataset_name, max_load_pct)


if __name__ == "__main__":
    init_logger(log_level="INFO")
    refresh()
