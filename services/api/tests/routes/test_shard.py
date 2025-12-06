# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Authors.

from typing import Any
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from api.routes.shard import create_shard_endpoint


@pytest.fixture
def client() -> TestClient:
    endpoint = create_shard_endpoint(
        hf_endpoint="https://huggingface.co",
        blocked_datasets=[],
        hf_token="token",
    )
    app = Starlette(routes=[Route("/shard", endpoint=endpoint)])
    return TestClient(app)


@pytest.fixture
def mock_cache_response() -> dict[str, Any]:
    """Standard multi-shard dataset response from config-parquet-and-info."""
    return {
        "content": {
            "dataset_info": {
                "splits": {
                    "train": {
                        "num_examples": 400,
                        "shard_lengths": [200, 200],
                        "original_shard_lengths": [100, 100, 100, 100],
                    }
                }
            },
            "parquet_files": [
                {
                    "filename": "train-00000-of-00002.parquet",
                    "split": "train",
                    "dataset": "test",
                    "config": "default",
                    "url": "...",
                    "size": 1000,
                },
                {
                    "filename": "train-00001-of-00002.parquet",
                    "split": "train",
                    "dataset": "test",
                    "config": "default",
                    "url": "...",
                    "size": 1000,
                },
            ],
            "partial": False,
        },
        "http_status": 200,
        "error_code": None,
        "dataset_git_revision": "abc123",
        "job_runner_version": 1,
        "progress": 1.0,
    }


@pytest.fixture
def mock_legacy_cache_response() -> dict[str, Any]:
    """Legacy dataset without original_shard_lengths."""
    return {
        "content": {
            "dataset_info": {
                "splits": {
                    "train": {
                        "num_examples": 400,
                        "shard_lengths": [200, 200],
                        # Note: NO original_shard_lengths key
                    }
                }
            },
            "parquet_files": [
                {
                    "filename": "train-00000-of-00002.parquet",
                    "split": "train",
                    "dataset": "test",
                    "config": "default",
                    "url": "...",
                    "size": 1000,
                },
                {
                    "filename": "train-00001-of-00002.parquet",
                    "split": "train",
                    "dataset": "test",
                    "config": "default",
                    "url": "...",
                    "size": 1000,
                },
            ],
            "partial": False,
        },
        "http_status": 200,
        "error_code": None,
        "dataset_git_revision": "abc123",
        "job_runner_version": 1,
        "progress": 1.0,
    }


def test_shard_endpoint_success(client: TestClient, mock_cache_response: dict[str, Any]) -> None:
    """GET /shard?dataset=X&config=Y&split=train&row=150 -> 200"""
    with patch("api.routes.shard.get_cache_entry_from_step", return_value=mock_cache_response):
        response = client.get("/shard?dataset=test&config=default&split=train&row=150")
        assert response.status_code == 200
        content = response.json()
        assert content["row_index"] == 150
        assert content["original_shard_index"] == 1
        assert content["parquet_shard_index"] == 0


def test_shard_endpoint_legacy_dataset(client: TestClient, mock_legacy_cache_response: dict[str, Any]) -> None:
    """Legacy dataset without original_shard_lengths -> graceful null"""
    with patch("api.routes.shard.get_cache_entry_from_step", return_value=mock_legacy_cache_response):
        response = client.get("/shard?dataset=test&config=default&split=train&row=150")
        assert response.status_code == 200
        content = response.json()
        assert content["original_shard_index"] is None
        assert content["original_shard_info"] is not None


def test_shard_endpoint_row_out_of_bounds(client: TestClient, mock_cache_response: dict[str, Any]) -> None:
    """row=500 when num_examples=400 -> 400 Bad Request"""
    with patch("api.routes.shard.get_cache_entry_from_step", return_value=mock_cache_response):
        response = client.get("/shard?dataset=test&config=default&split=train&row=500")
        assert response.status_code == 400
        assert response.headers.get("X-Error-Code") == "RowOutOfBounds"


def test_shard_endpoint_missing_params(client: TestClient) -> None:
    """Missing required parameters -> 422"""
    response = client.get("/shard?dataset=test")  # Missing config, split, row
    assert response.status_code == 422
