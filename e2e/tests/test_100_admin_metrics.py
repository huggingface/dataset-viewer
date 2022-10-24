# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os


from .fixtures.hub import DatasetRepos
from .utils import get


def test_is_valid_after_datasets_processed(hf_dataset_repos_csv_data: DatasetRepos):
    ADMIN_UVICORN_PORT = os.environ.get("ADMIN_UVICORN_PORT", "10000")
    URL = f"http://localhost:{ADMIN_UVICORN_PORT}"
    is_multiprocess = "PROMETHEUS_MULTIPROC_DIR" in os.environ
    assert is_multiprocess
    response = get("/metrics", url=URL)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    content = response.text
    lines = content.split("\n")
    metrics = {line.split(" ")[0]: float(line.split(" ")[1]) for line in lines if line and line[0] != "#"}
    if not is_multiprocess:
        name = "process_start_time_seconds"
        assert name in metrics
        assert metrics[name] > 0
    additional_field = ('pid="' + str(os.getpid()) + '",') if is_multiprocess else ""
    for queue in ["/splits", "/first-rows"]:
        assert "queue_jobs_total{" + additional_field + 'queue="' + queue + '",status="started"}' in metrics
    assert (
        "responses_in_cache_total{" + additional_field + 'path="/splits",http_status="200",error_code=null}'
        not in metrics
    )
    assert (
        "responses_in_cache_total{" + additional_field + 'path="/first-rows",http_status="200",error_code=null}'
        not in metrics
    )
