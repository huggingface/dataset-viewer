from prometheus_client import start_http_server

from api.config import METRICS_PORT


def start_metrics() -> None:
    start_http_server(METRICS_PORT)
