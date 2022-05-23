import os
from typing import Dict

from libcache.cache import get_datasets_count_by_status, get_splits_count_by_status
from libqueue.queue import (
    get_dataset_jobs_count_by_status,
    get_split_jobs_count_by_status,
)
from prometheus_client import (  # type: ignore # https://github.com/prometheus/client_python/issues/491
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Gauge,
    generate_latest,
)
from prometheus_client.multiprocess import (  # type: ignore # https://github.com/prometheus/client_python/issues/491
    MultiProcessCollector,
)
from starlette.requests import Request
from starlette.responses import Response


class Prometheus:
    metrics: Dict[str, Gauge] = {}

    def __init__(self):
        self.initMetrics()

    def getRegistry(self) -> CollectorRegistry:
        # taken from https://github.com/perdy/starlette-prometheus/blob/master/starlette_prometheus/view.py
        if "prometheus_multiproc_dir" in os.environ:
            registry = CollectorRegistry()
            MultiProcessCollector(registry)
        else:
            registry = REGISTRY
        return registry

    # add metrics from the databases
    def initMetrics(self):
        self.metrics["queue_jobs_total"] = Gauge(
            "queue_jobs_total", "Number of jobs in the queue", ["queue", "status"]
        )
        self.metrics["cache_entries_total"] = Gauge(
            "cache_entries_total", "Number of entries in the cache", ["cache", "status"]
        )

    def updateMetrics(self):
        for status, total in get_dataset_jobs_count_by_status().items():
            self.metrics["queue_jobs_total"].labels(queue="datasets", status=status).set(total)
        for status, total in get_split_jobs_count_by_status().items():
            self.metrics["queue_jobs_total"].labels(queue="splits", status=status).set(total)
        for status, total in get_datasets_count_by_status().items():
            self.metrics["cache_entries_total"].labels(cache="datasets", status=status).set(total)
        for status, total in get_splits_count_by_status().items():
            self.metrics["cache_entries_total"].labels(cache="splits", status=status).set(total)

    def endpoint(self, request: Request) -> Response:
        self.updateMetrics()

        return Response(generate_latest(self.getRegistry()), headers={"Content-Type": CONTENT_TYPE_LATEST})
