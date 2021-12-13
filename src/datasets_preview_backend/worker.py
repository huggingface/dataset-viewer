import logging
import os
import time

from dotenv import load_dotenv
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from datasets_preview_backend.constants import (
    DEFAULT_MAX_LOAD_PCT,
    DEFAULT_MAX_MEMORY_PCT,
    DEFAULT_WORKER_SLEEP_SECONDS,
)
from datasets_preview_backend.io.cache import connect_to_cache, refresh_dataset
from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import (
    EmptyQueue,
    connect_to_queue,
    finish_job,
    get_job,
)
from datasets_preview_backend.utils import get_int_value

# Load environment variables defined in .env, if any
load_dotenv()
max_load_pct = get_int_value(os.environ, "MAX_LOAD_PCT", DEFAULT_MAX_LOAD_PCT)
max_memory_pct = get_int_value(os.environ, "MAX_MEMORY_PCT", DEFAULT_MAX_MEMORY_PCT)
worker_sleep_seconds = get_int_value(os.environ, "WORKER_SLEEP_SECONDS", DEFAULT_WORKER_SLEEP_SECONDS)


def process_next_job() -> bool:
    logger = logging.getLogger("worker")
    logger.debug("try to process a job")

    try:
        job_id, dataset_name = get_job()
        logger.debug(f"job assigned: {job_id} for dataset: {dataset_name}")
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    try:
        logger.info(f"compute dataset '{dataset_name}'")
        refresh_dataset(dataset_name=dataset_name)
    finally:
        finish_job(job_id)
        logger.debug(f"job finished: {job_id} for dataset: {dataset_name}")
    return True


def has_memory() -> bool:
    logger = logging.getLogger("worker")
    virtual_memory_used: int = virtual_memory().used  # type: ignore
    virtual_memory_total: int = virtual_memory().total  # type: ignore
    percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
    ok = percent < max_memory_pct
    if not ok:
        logger.info(f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is {max_memory_pct}%")
    return ok


def has_cpu() -> bool:
    logger = logging.getLogger("worker")
    load_pct = max(x / cpu_count() * 100 for x in getloadavg())
    ok = load_pct < max_load_pct
    if not ok:
        logger.info(f"cpu load is too high: {load_pct:.0f}% - max is {max_load_pct}%")
    return ok


def has_resources() -> bool:
    return has_memory() and has_cpu()


def sleep() -> None:
    logger = logging.getLogger("worker")
    logger.debug(f"sleep during {worker_sleep_seconds} seconds")
    time.sleep(worker_sleep_seconds)


def loop() -> None:
    while not has_resources() or not process_next_job():
        sleep()
    # a job has been processed - exit
    # the worker should be restarted automatically by pm2
    # this way, we avoid using too much RAM+SWAP


if __name__ == "__main__":
    init_logger("DEBUG", "worker")
    connect_to_cache()
    connect_to_queue()
    loop()
