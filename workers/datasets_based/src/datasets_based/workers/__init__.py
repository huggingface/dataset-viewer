# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List, Type

from datasets_based.workers._datasets_based_worker import DatasetsBasedWorker
from datasets_based.workers.first_rows import FirstRowsWorker
from datasets_based.workers.parquet import ParquetWorker
from datasets_based.workers.splits import SplitsWorker

worker_classes: List[Type[DatasetsBasedWorker]] = [FirstRowsWorker, ParquetWorker, SplitsWorker]
worker_class_by_endpoint = {worker_class.get_endpoint(): worker_class for worker_class in worker_classes}

# explicit re-export
__all__ = ["DatasetsBasedWorker"]
