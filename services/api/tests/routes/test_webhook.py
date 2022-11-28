# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Any, Mapping

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
