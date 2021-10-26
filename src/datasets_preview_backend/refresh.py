import logging
import os
from random import random
from typing import Callable

from dotenv import load_dotenv

from datasets_preview_backend.constants import DEFAULT_REFRESH_PCT
from datasets_preview_backend.io.cache import connect_to_cache
from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.queue import add_job, connect_to_queue
from datasets_preview_backend.models.hf_dataset import get_hf_dataset_names
from datasets_preview_backend.utils import get_int_value

# Load environment variables defined in .env, if any
load_dotenv()


# TODO: we could get the first N, sorted by creation time (more or less expire time)
def make_criterion(threshold: float) -> Callable[[str], bool]:
    def criterion(_: str) -> bool:
        return random() < threshold  # nosec

    return criterion


def refresh() -> None:
    logger = logging.getLogger("refresh")
    refresh_pct = get_int_value(os.environ, "REFRESH_PCT", DEFAULT_REFRESH_PCT)
    dataset_names = get_hf_dataset_names()
    criterion = make_criterion(refresh_pct / 100)
    selected_dataset_names = list(filter(criterion, dataset_names))
    logger.info(
        f"Refreshing: {len(selected_dataset_names)} datasets from"
        f" {len(dataset_names)} ({100*len(selected_dataset_names)/len(dataset_names):.1f}%)"
    )

    for dataset_name in selected_dataset_names:
        add_job(dataset_name)
        logger.info(f"added a job to refresh '{dataset_name}'")


if __name__ == "__main__":
    init_logger("INFO", "refresh")
    connect_to_cache()
    connect_to_queue()
    refresh()
