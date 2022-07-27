import logging

from libqueue.queue import (
    EmptyQueue,
    add_dataset_job,
    add_split_job,
    finish_dataset_job,
    finish_split_job,
    get_dataset_job,
    get_split_job,
)
from libutils.exceptions import Status500Error, StatusError

from worker.config import (
    HF_TOKEN,
    MAX_JOB_RETRIES,
    MAX_JOBS_PER_DATASET,
    MAX_SIZE_FALLBACK,
    ROWS_MAX_BYTES,
    ROWS_MAX_NUMBER,
    ROWS_MIN_NUMBER,
)
from worker.deprecated.refresh import refresh_dataset, refresh_split


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
