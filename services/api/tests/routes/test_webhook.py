# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Any

import pytest

from api.routes.webhook import parse_payload


@pytest.mark.parametrize(
    "payload,raises",
    [
        ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}}, False),
        (
            {
                "event": "move",
                "movedTo": "webhook-test",
                "repo": {"type": "dataset", "name": "previous-name", "gitalyUid": "123"},
            },
            False,
        ),
        ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test"}}, False),
        ({"event": "doesnotexist", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}}, True),
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
            {"event": "update", "repo": {"type": "dataset", "name": "AresEkb/prof_standards_sbert_large_mt_nlu_ru"}},
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
