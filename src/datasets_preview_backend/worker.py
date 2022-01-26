import logging
import os
import time

from dotenv import load_dotenv
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from datasets_preview_backend.constants import (
    DEFAULT_DATASETS_REVISION,
    DEFAULT_HF_TOKEN,
    DEFAULT_MAX_LOAD_PCT,
    DEFAULT_MAX_MEMORY_PCT,
    DEFAULT_MAX_SIZE_FALLBACK,
    DEFAULT_WORKER_QUEUE,
    DEFAULT_WORKER_SLEEP_SECONDS,
)
from datasets_preview_backend.io.cache import (
    connect_to_cache,
    refresh_dataset_split_full_names,
    refresh_split,
)
from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import (
    EmptyQueue,
    add_split_job,
    connect_to_queue,
    finish_dataset_job,
    finish_split_job,
    get_dataset_job,
    get_split_job,
)
from datasets_preview_backend.utils import (
    get_int_value,
    get_str_or_none_value,
    get_str_value,
)

# Load environment variables defined in .env, if any
load_dotenv()
max_load_pct = get_int_value(os.environ, "MAX_LOAD_PCT", DEFAULT_MAX_LOAD_PCT)
max_memory_pct = get_int_value(os.environ, "MAX_MEMORY_PCT", DEFAULT_MAX_MEMORY_PCT)
worker_sleep_seconds = get_int_value(os.environ, "WORKER_SLEEP_SECONDS", DEFAULT_WORKER_SLEEP_SECONDS)
hf_token = get_str_or_none_value(d=os.environ, key="HF_TOKEN", default=DEFAULT_HF_TOKEN)
max_size_fallback = get_int_value(os.environ, "MAX_SIZE_FALLBACK", DEFAULT_MAX_SIZE_FALLBACK)
worker_queue = get_str_value(os.environ, "WORKER_QUEUE", DEFAULT_WORKER_QUEUE)
# Ensure datasets library uses the expected revision for canonical datasets
os.environ["HF_SCRIPTS_VERSION"] = get_str_value(
    d=os.environ, key="DATASETS_REVISION", default=DEFAULT_DATASETS_REVISION
)


def process_next_dataset_job() -> bool:
    logger = logging.getLogger("datasets_preview_backend.worker")
    logger.debug("try to process a dataset job")

    try:
        job_id, dataset_name = get_dataset_job()
        logger.debug(f"job assigned: {job_id} for dataset: {dataset_name}")
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    try:
        logger.info(f"compute dataset '{dataset_name}'")
        split_full_names = refresh_dataset_split_full_names(dataset_name=dataset_name, hf_token=hf_token)
        for split_full_name in split_full_names:
            add_split_job(
                split_full_name["dataset_name"], split_full_name["config_name"], split_full_name["split_name"]
            )
        finish_dataset_job(job_id, success=True)
        logger.debug(f"job finished with success: {job_id} for dataset: {dataset_name}")
    except Exception:
        finish_dataset_job(job_id, success=False)
        logger.debug(f"job finished with error: {job_id} for dataset: {dataset_name}")
    return True


def process_next_split_job() -> bool:
    logger = logging.getLogger("datasets_preview_backend.worker")
    logger.debug("try to process a split job")

    try:
        job_id, dataset_name, config_name, split_name = get_split_job()
        logger.debug(
            f"job assigned: {job_id} for split '{split_name}' from dataset '{dataset_name}' with config"
            f" '{config_name}'"
        )
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    try:
        logger.info(f"compute split '{split_name}' from dataset '{dataset_name}' with config '{config_name}'")
        refresh_split(
            dataset_name=dataset_name,
            config_name=config_name,
            split_name=split_name,
            hf_token=hf_token,
            max_size_fallback=max_size_fallback,
        )
        finish_split_job(job_id, success=True)
        logger.debug(
            f"job finished with success: {job_id} for split '{split_name}' from dataset '{dataset_name}' with config"
            f" '{config_name}'"
        )
    except Exception:
        finish_split_job(job_id, success=False)
        logger.debug(
            f"job finished with error: {job_id} for split '{split_name}' from dataset '{dataset_name}' with config"
            f" '{config_name}'"
        )
    return True


def process_next_job() -> bool:
    if worker_queue == "datasets":
        return process_next_dataset_job()
    elif worker_queue == "splits":
        return process_next_split_job()
    raise NotImplementedError(f"Job queue {worker_queue} does not exist")


def has_memory() -> bool:
    logger = logging.getLogger("datasets_preview_backend.worker")
    virtual_memory_used: int = virtual_memory().used  # type: ignore
    virtual_memory_total: int = virtual_memory().total  # type: ignore
    percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
    ok = percent < max_memory_pct
    if not ok:
        logger.info(f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is {max_memory_pct}%")
    return ok


def has_cpu() -> bool:
    logger = logging.getLogger("datasets_preview_backend.worker")
    load_pct = max(getloadavg()[:2]) / cpu_count() * 100
    # ^ only current load and 5m load. 15m load is not relevant to decide to launch a new job
    ok = load_pct < max_load_pct
    if not ok:
        logger.info(f"cpu load is too high: {load_pct:.0f}% - max is {max_load_pct}%")
    return ok


def has_resources() -> bool:
    return has_memory() and has_cpu()


def sleep() -> None:
    logger = logging.getLogger("datasets_preview_backend.worker")
    logger.debug(f"sleep during {worker_sleep_seconds} seconds")
    time.sleep(worker_sleep_seconds)


def loop() -> None:
    while not has_resources() or not process_next_job():
        sleep()
    # a job has been processed - exit
    # the worker should be restarted automatically by pm2
    # this way, we avoid using too much RAM+SWAP


if __name__ == "__main__":
    init_logger("DEBUG")
    connect_to_cache()
    connect_to_queue()
    loop()
