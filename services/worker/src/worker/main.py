# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
import time
from http import HTTPStatus

from libcache.asset import show_assets_dir
from libcache.simple_cache import connect_to_cache
from libqueue.queue import EmptyQueue, connect_to_queue
from libutils.logger import init_logger
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from .config import (
    ASSETS_BASE_URL,
    ASSETS_DIRECTORY,
    HF_ENDPOINT,
    HF_TOKEN,
    LOG_LEVEL,
    MAX_JOBS_PER_DATASET,
    MAX_LOAD_PCT,
    MAX_MEMORY_PCT,
    MAX_SIZE_FALLBACK,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
    ROWS_MAX_BYTES,
    ROWS_MAX_NUMBER,
    ROWS_MIN_NUMBER,
    WORKER_QUEUE,
    WORKER_SLEEP_SECONDS,
)
from .refresh import refresh_first_rows, refresh_splits
from .utils import Queues


def process_next_splits_job(queues: Queues) -> bool:
    logger = logging.getLogger("datasets_server.worker")
    logger.debug("try to process a splits/ job")

    try:
        job_id, dataset, *_ = queues.splits.start_job()
        logger.debug(f"job assigned: {job_id} for dataset={dataset}")
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    success = False
    try:
        logger.info(f"compute dataset={dataset}")
        http_status = refresh_splits(queues, dataset=dataset, hf_endpoint=HF_ENDPOINT, hf_token=HF_TOKEN)
        success = http_status == HTTPStatus.OK
    finally:
        queues.splits.finish_job(job_id=job_id, success=success)
        result = "success" if success else "error"
        logger.debug(f"job finished with {result}: {job_id} for dataset={dataset}")
    return True


def process_next_first_rows_job(queues: Queues) -> bool:
    logger = logging.getLogger("datasets_server.worker")
    logger.debug("try to process a first-rows job")

    try:
        job_id, dataset, config, split = queues.first_rows.start_job()
        logger.debug(f"job assigned: {job_id} for dataset={dataset} config={config} split={split}")
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    success = False
    try:
        logger.info(f"compute dataset={dataset} config={config} split={split}")
        if config is None or split is None:
            raise ValueError("config and split are required")
        http_status = refresh_first_rows(
            dataset=dataset,
            config=config,
            split=split,
            assets_base_url=ASSETS_BASE_URL,
            hf_endpoint=HF_ENDPOINT,
            hf_token=HF_TOKEN,
            max_size_fallback=MAX_SIZE_FALLBACK,
            rows_max_bytes=ROWS_MAX_BYTES,
            rows_max_number=ROWS_MAX_NUMBER,
            rows_min_number=ROWS_MIN_NUMBER,
        )
        success = http_status == HTTPStatus.OK
    finally:
        queues.first_rows.finish_job(job_id=job_id, success=success)
        result = "success" if success else "error"
        logger.debug(f"job finished with {result}: {job_id} for dataset={dataset} config={config} split={split}")
    return True


def process_next_job(queues: Queues) -> bool:
    if WORKER_QUEUE == "first_rows_responses":
        return process_next_first_rows_job(queues)
    elif WORKER_QUEUE == "splits_responses":
        return process_next_splits_job(queues)
    raise NotImplementedError(f"Job queue {WORKER_QUEUE} does not exist")


def has_memory() -> bool:
    logger = logging.getLogger("datasets_server.worker")
    if MAX_MEMORY_PCT <= 0:
        return True
    virtual_memory_used: int = virtual_memory().used  # type: ignore
    virtual_memory_total: int = virtual_memory().total  # type: ignore
    percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
    ok = percent < MAX_MEMORY_PCT
    if not ok:
        logger.info(f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is {MAX_MEMORY_PCT}%")
    return ok


def has_cpu() -> bool:
    logger = logging.getLogger("datasets_server.worker")
    if MAX_LOAD_PCT <= 0:
        return True
    load_pct = max(getloadavg()[:2]) / cpu_count() * 100
    # ^ only current load and 5m load. 15m load is not relevant to decide to launch a new job
    ok = load_pct < MAX_LOAD_PCT
    if not ok:
        logger.info(f"cpu load is too high: {load_pct:.0f}% - max is {MAX_LOAD_PCT}%")
    return ok


def has_resources() -> bool:
    return has_memory() and has_cpu()


def sleep() -> None:
    logger = logging.getLogger("datasets_server.worker")
    jitter = 0.75 + random.random() / 2  # nosec
    # ^ between 0.75 and 1.25
    duration = WORKER_SLEEP_SECONDS * jitter
    logger.debug(f"sleep during {duration:.2f} seconds")
    time.sleep(duration)


def loop(queues: Queues) -> None:
    logger = logging.getLogger("datasets_server.worker")
    try:
        while True:
            if has_resources() and process_next_job(queues):
                # loop immediately to try another job
                # see https://github.com/huggingface/datasets-server/issues/265
                continue
            sleep()
    except BaseException as e:
        logger.critical(f"quit due to an uncaught error while processing the job: {e}")
        raise


if __name__ == "__main__":
    init_logger(LOG_LEVEL)
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)
    show_assets_dir(ASSETS_DIRECTORY)
    loop(queues=Queues(max_jobs_per_dataset=MAX_JOBS_PER_DATASET))
