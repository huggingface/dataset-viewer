# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest

from webhook.routes.webhook import MoonWebhookV2Payload, parse_payload, process_payload


@pytest.mark.parametrize(
    "payload,raises",
    [
        (
            {"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}, "scope": "repo"},
            False,
        ),
        (
            {
                "event": "move",
                "movedTo": "webhook-test",
                "repo": {"type": "dataset", "name": "previous-name", "gitalyUid": "123"},
                "scope": "repo",
            },
            False,
        ),
        ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test"}, "scope": "repo"}, False),
        (
            {
                "event": "doesnotexist",
                "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"},
                "scope": "repo",
            },
            True,
        ),
        (
            {
                "event": "update",
                "scope": "repo.content",
                "repo": {
                    "type": "dataset",
                    "name": "AresEkb/prof_standards_sbert_large_mt_nlu_ru",
                    "id": "63bab13ae0f4fee16cebf084",
                    "private": False,
                    "url": {
                        "web": "https://huggingface.co/datasets/AresEkb/prof_standards_sbert_large_mt_nlu_ru",
                        "api": "https://huggingface.co/api/datasets/AresEkb/prof_standards_sbert_large_mt_nlu_ru",
                    },
                    "headSha": "c926e6ce93cbd5a6eaf0895abd48776cc5bae638",
                    "gitalyUid": "c5afeca93171cfa1f6c138ef683df4a53acffd8c86283ab8e7e338df369d2fff",
                    "authorId": "6394b8740b746ac6a969bd51",
                    "tags": [],
                },
                "webhook": {"id": "632c22b3df82fca9e3b46154", "version": 2},
            },
            False,
        ),
        (
            {
                "event": "update",
                "repo": {"type": "dataset", "name": "AresEkb/prof_standards_sbert_large_mt_nlu_ru", "headSha": "abc"},
                "scope": "repo.content",
            },
            False,
        ),
    ],
)
def test_parse_payload(
    payload: Mapping[str, Any],
    raises: bool,
) -> None:
    if raises:
        with pytest.raises(Exception):
            parse_payload(payload)
    else:
        parse_payload(payload)


@pytest.mark.parametrize(
    "payload,does_update",
    [
        (
            {"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}, "scope": "repo"},
            True,
        ),
        (
            {
                "event": "move",
                "movedTo": "webhook-test",
                "repo": {"type": "dataset", "name": "previous-name", "gitalyUid": "123"},
                "scope": "repo",
            },
            True,
        ),
        ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test"}, "scope": "repo"}, True),
        (
            {
                "event": "doesnotexist",
                "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"},
                "scope": "repo",
            },
            False,
        ),
        (
            {
                "event": "update",
                "scope": "repo.content",
                "repo": {
                    "type": "dataset",
                    "name": "AresEkb/prof_standards_sbert_large_mt_nlu_ru",
                    "id": "63bab13ae0f4fee16cebf084",
                    "private": False,
                    "url": {
                        "web": "https://huggingface.co/datasets/AresEkb/prof_standards_sbert_large_mt_nlu_ru",
                        "api": "https://huggingface.co/api/datasets/AresEkb/prof_standards_sbert_large_mt_nlu_ru",
                    },
                    "headSha": "c926e6ce93cbd5a6eaf0895abd48776cc5bae638",
                    "gitalyUid": "c5afeca93171cfa1f6c138ef683df4a53acffd8c86283ab8e7e338df369d2fff",
                    "authorId": "6394b8740b746ac6a969bd51",
                    "tags": [],
                },
                "webhook": {"id": "632c22b3df82fca9e3b46154", "version": 2},
            },
            True,
        ),
        (
            {
                "event": "update",
                "repo": {"type": "dataset", "name": "AresEkb/prof_standards_sbert_large_mt_nlu_ru", "headSha": "abc"},
                "scope": "repo.content",
            },
            True,
        ),
        (
            {
                "event": "add",
                "scope": "discussion.comment",
                "repo": {
                    "type": "dataset",
                    "name": "allenai/c4",
                    "id": "621ffdd236468d709f182a80",
                    "private": False,
                    "url": {
                        "web": "https://huggingface.co/datasets/allenai/c4",
                        "api": "https://huggingface.co/api/datasets/allenai/c4",
                    },
                    "gitalyUid": "efe8f46938ffc097e3451ddd714b6e95d2aed3bf0b33ee13b48674103ae98292",
                    "authorId": "5e70f3648ce3c604d78fe132",
                    "tags": [],
                },
                "discussion": {
                    "id": "659e0e0f1723f371c9ee8745",
                    "title": "Cannot load dataset from hf hub",
                    "url": {
                        "web": "https://huggingface.co/datasets/allenai/c4/discussions/7",
                        "api": "https://huggingface.co/api/datasets/allenai/c4/discussions/7",
                    },
                    "status": "open",
                    "author": {"id": "649e3f263914db6cf8e8ab1f"},
                    "num": 7,
                    "isPullRequest": False,
                },
                "comment": {
                    "id": "659ec0352ede35f008a65c2b",
                    "author": {"id": "5e9ecfc04957053f60648a3e"},
                    "content": 'Hi ! The \'allenai--c4\' configuration doesn"t exist for this dataset (it"s a legacy scheme from old versions of the `datasets` library)\n\nYou can try this instead:\n\n```python\nfrom datasets import load_dataset\ntraindata = load_dataset(\n "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"\n)\nvaldata = load_dataset(\n "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"\n)\n```',
                    "hidden": False,
                    "url": {
                        "web": "https://huggingface.co/datasets/allenai/c4/discussions/7#659ec0352ede35f008a65c2b"
                    },
                },
                "webhook": {"id": "632c22b3df82fca9e3b46154", "version": 2},
            },
            False,
        ),
    ],
)
def test_process_payload(
    payload: MoonWebhookV2Payload,
    does_update: bool,
) -> None:
    with patch("api.routes.webhook.delete_dataset") as mock_delete_dataset:
        with patch("api.routes.webhook.update_dataset") as mock_update_dataset:
            process_payload(payload, blocked_datasets=[], hf_endpoint="https://huggingface.co")
            assert mock_delete_dataset.call_count == int(does_update)
            assert mock_update_dataset.call_count == int(does_update)
