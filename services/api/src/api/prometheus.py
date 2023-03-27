# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
import time
from types import TracebackType
from typing import Any, Optional, Type, TypeVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Histogram,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector

# ^ type: ignore can be removed on next release:
# https://github.com/prometheus/client_python/issues/491#issuecomment-1429287314
from starlette.requests import Request
from starlette.responses import Response

# the metrics are global to the process
METHOD_STEPS_PROCESSING_TIME = Histogram(
    "method_steps_processing_time_seconds",
    "Histogram of the processing time of specific steps in methods for a given context (in seconds)",
    ["method", "step", "context"],
)

T = TypeVar("T", bound="StepProfiler")


class StepProfiler:
    """
    A context manager that measures the time spent in a step of a method and reports it to Prometheus.

    Example:
        >>> with StepProfiler("method", "step", "context") as profiler:
        ...     pass

    Args:
        method (str): The name of the method.
        step (str): The name of the step.
        context (str|None): An optional string that adds context. If None, the label "None" is used.
    """

    def __init__(self, method: str, step: str, context: Optional[str] = None):
        self.method = method
        self.step = step
        self.context = str(context)
        self.before_time = time.perf_counter()

    def __enter__(self: T) -> T:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        after_time = time.perf_counter()
        METHOD_STEPS_PROCESSING_TIME.labels(method=self.method, step=self.step, context=self.context).observe(
            after_time - self.before_time
        )


class Prometheus:
    def getRegistry(self) -> CollectorRegistry:
        # taken from https://github.com/perdy/starlette-prometheus/blob/master/starlette_prometheus/view.py
        # see https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            registry = CollectorRegistry()
            MultiProcessCollector(registry=registry)
        else:
            registry = REGISTRY
        return registry

    def getLatestContent(self) -> Any:
        # ^ returns Any because we cannot be sure latest are UTF8Bytes
        latest = generate_latest(self.getRegistry())
        return latest.decode("utf-8")

    def endpoint(self, request: Request) -> Response:
        return Response(self.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})
