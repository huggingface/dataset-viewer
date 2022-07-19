import logging
import random
import time

from libcache.asset import show_assets_dir
from libcache.cache import connect_to_cache
from libcache.simple_cache import HTTPStatus
from libqueue.queue import (
    EmptyQueue,
    add_dataset_job,
    add_first_rows_job,
    add_split_job,
    add_splits_job,
    connect_to_queue,
    finish_dataset_job,
    finish_first_rows_job,
    finish_split_job,
    finish_splits_job,
    get_dataset_job,
    get_first_rows_job,
    get_split_job,
    get_splits_job,
)
from libutils.exceptions import Status500Error, StatusError
from libutils.logger import init_logger
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from worker.config import (
    ASSETS_BASE_URL,
    ASSETS_DIRECTORY,
    HF_TOKEN,
    LOG_LEVEL,
    MAX_JOB_RETRIES,
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
from worker.refresh import (
    refresh_dataset,
    refresh_first_rows,
    refresh_split,
    refresh_splits,
)


def process_next_dataset_job() -> bool:
    logger = logging.getLogger("datasets_server.worker")
    logger.debug("try to process a dataset job")

    try:
        job_id, dataset_name, retries = get_dataset_job(MAX_JOBS_PER_DATASET)
        logger.debug(f"job assigned: {job_id} for dataset={dataset_name}")
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    success = False
    retry = False
    try:
        logger.info(f"compute dataset={dataset_name}")
        refresh_dataset(dataset_name=dataset_name, hf_token=HF_TOKEN)
        success = True
    except StatusError as e:
        if isinstance(e, Status500Error) and retries < MAX_JOB_RETRIES:
            retry = True
        # in any case: don't raise the StatusError, and go to finally
    finally:
        finish_dataset_job(job_id, success=success)
        result = "success" if success else "error"
        logger.debug(f"job finished with {result}: {job_id} for dataset={dataset_name}")
        if retry:
            add_dataset_job(dataset_name, retries=retries + 1)
            logger.debug(f"job re-enqueued (retries: {retries}) for dataset={dataset_name}")
    return True


def process_next_split_job() -> bool:
    logger = logging.getLogger("datasets_server.worker")
    logger.debug("try to process a split job")

    try:
        job_id, dataset_name, config_name, split_name, retries = get_split_job(MAX_JOBS_PER_DATASET)
        logger.debug(f"job assigned: {job_id} for dataset={dataset_name} config={config_name} split={split_name}")
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    success = False
    retry = False
    try:
        logger.info(f"compute dataset={dataset_name} config={config_name} split={split_name}")
        refresh_split(
            dataset_name=dataset_name,
            config_name=config_name,
            split_name=split_name,
            hf_token=HF_TOKEN,
            max_size_fallback=MAX_SIZE_FALLBACK,
            rows_max_bytes=ROWS_MAX_BYTES,
            rows_max_number=ROWS_MAX_NUMBER,
            rows_min_number=ROWS_MIN_NUMBER,
        )
        success = True
    except StatusError as e:
        if isinstance(e, Status500Error) and retries < MAX_JOB_RETRIES:
            retry = True
        # in any case: don't raise the StatusError, and go to finally
    finally:
        finish_split_job(job_id, success=success)
        result = "success" if success else "error"
        logger.debug(
            f"job finished with {result}: {job_id} for dataset={dataset_name} config={config_name} split={split_name}"
        )
        if retry:
            add_split_job(dataset_name, config_name, split_name, retries=retries + 1)
            logger.debug(
                f"job re-enqueued (retries: {retries}) for"
                f" dataset={dataset_name} config={config_name} split={split_name}"
            )
    return True


def process_next_splits_job() -> bool:
    logger = logging.getLogger("datasets_server.worker")
    logger.debug("try to process a splits/ job")

    try:
        job_id, dataset_name, retries = get_splits_job(MAX_JOBS_PER_DATASET)
        logger.debug(f"job assigned: {job_id} for dataset={dataset_name}")
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    success = False
    retry = False
    try:
        logger.info(f"compute dataset={dataset_name}")
        http_status = refresh_splits(dataset_name=dataset_name, hf_token=HF_TOKEN)
        success = http_status == HTTPStatus.OK
        if http_status == HTTPStatus.INTERNAL_SERVER_ERROR and retries < MAX_JOB_RETRIES:
            retry = True
    finally:
        finish_splits_job(job_id, success=success)
        result = "success" if success else "error"
        logger.debug(f"job finished with {result}: {job_id} for dataset={dataset_name}")
        if retry:
            add_splits_job(dataset_name, retries=retries + 1)
            logger.debug(f"job re-enqueued (retries: {retries}) for dataset={dataset_name}")
    return True


def process_next_first_rows_job() -> bool:
    logger = logging.getLogger("datasets_server.worker")
    logger.debug("try to process a first-rows job")

    try:
        job_id, dataset_name, config_name, split_name, retries = get_first_rows_job(MAX_JOBS_PER_DATASET)
        logger.debug(f"job assigned: {job_id} for dataset={dataset_name} config={config_name} split={split_name}")
    except EmptyQueue:
        logger.debug("no job in the queue")
        return False

    success = False
    retry = False
    try:
        logger.info(f"compute dataset={dataset_name} config={config_name} split={split_name}")
        http_status = refresh_first_rows(
            dataset_name=dataset_name,
            config_name=config_name,
            split_name=split_name,
            assets_base_url=ASSETS_BASE_URL,
            hf_token=HF_TOKEN,
            max_size_fallback=MAX_SIZE_FALLBACK,
            rows_max_bytes=ROWS_MAX_BYTES,
            rows_max_number=ROWS_MAX_NUMBER,
            rows_min_number=ROWS_MIN_NUMBER,
        )
        success = http_status == HTTPStatus.OK
        if http_status == HTTPStatus.INTERNAL_SERVER_ERROR and retries < MAX_JOB_RETRIES:
            retry = True
    finally:
        finish_first_rows_job(job_id, success=success)
        result = "success" if success else "error"
        logger.debug(
            f"job finished with {result}: {job_id} for dataset={dataset_name} config={config_name} split={split_name}"
        )
        if retry:
            add_first_rows_job(dataset_name, config_name, split_name, retries=retries + 1)
            logger.debug(
                f"job re-enqueued (retries: {retries}) for"
                f" dataset={dataset_name} config={config_name} split={split_name}"
            )
    return True


def process_next_job() -> bool:
    if WORKER_QUEUE == "datasets":
        return process_next_dataset_job()
    elif WORKER_QUEUE == "first_rows_responses":
        return process_next_first_rows_job()
    elif WORKER_QUEUE == "splits":
        return process_next_split_job()
    elif WORKER_QUEUE == "splits_responses":
        return process_next_splits_job()
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


def loop() -> None:
    logger = logging.getLogger("datasets_server.worker")
    try:
        while True:
            if has_resources() and process_next_job():
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
    loop()
