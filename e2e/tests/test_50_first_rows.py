# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
from typing import Any

import pytest
import requests

from .utils import (
    URL,
    get,
    get_openapi_body_example,
    poll,
    refresh_poll_splits,
    refresh_poll_splits_first_rows,
)


def prepare_json(response: requests.Response) -> Any:
    return json.loads(response.text.replace(URL, "https://datasets-server.huggingface.co"))


@pytest.mark.parametrize(
    "status,name,dataset,config,split,error_code",
    [
        # (200, "imdb", "imdb", "plain_text", "train", None),
        # (200, "truncated", "ett", "m2", "test", None),
        # (200, "image", "huggan/horse2zebra", "huggan--horse2zebra-aligned", "train", None),
        # (200, "audio", "mozilla-foundation/common_voice_9_0", "en", "train", None),
        # ^ awfully long
        (
            401,
            "inexistent-dataset",
            "severo/inexistent-dataset",
            "plain_text",
            "train",
            "ExternalUnauthenticatedError",
        ),
        # (
        #     401,
        #     "gated-dataset",
        #     "severo/dummy_gated",
        #     "severo--embellishments",
        #     "train",
        #     "ExternalUnauthenticatedError",
        # ),
        # (
        #     401,
        #     "private-dataset",
        #     "severo/dummy_private",
        #     "severo--embellishments",
        #     "train",
        #     "ExternalUnauthenticatedError",
        # ),
        # (404, "inexistent-config", "imdb", "inexistent-config", "train", "FirstRowsResponseNotFound"),
        # (404, "inexistent-split", "imdb", "plain_text", "inexistent-split", "FirstRowsResponseNotFound"),
        (422, "missing-dataset", None, "plain_text", "train", "MissingRequiredParameter"),
        (422, "missing-config", "imdb", None, "train", "MissingRequiredParameter"),
        (422, "missing-split", "imdb", "plain_text", None, "MissingRequiredParameter"),
        (422, "empty-dataset", "", "plain_text", "train", "MissingRequiredParameter"),
        (422, "empty-config", "imdb", "", "train", "MissingRequiredParameter"),
        (422, "empty-split", "imdb", "plain_text", "", "MissingRequiredParameter"),
        # (500, "NonMatchingCheckError", "ar_cov19", "ar_cov19", "train", "NormalRowsError"),
        # (500, "FileNotFoundError", "atomic", "atomic", "train", "NormalRowsError"),
        # (500, "not-ready", "anli", "plain_text", "train_r1", "FirstRowsResponseNotReady"),
        # not tested: 'internal_error'
        # TODO:
        # "SplitsNamesError",
        # "InfoError",
        # "FeaturesError",
        # "StreamingRowsError",
        # "RowsPostProcessingError",
    ],
)
def test_first_rows(status: int, name: str, dataset: str, config: str, split: str, error_code: str):
    body = get_openapi_body_example("/first-rows", status, name)

    # the logic here is a bit convoluted, because we have no way to refresh a split, we have to refresh the whole
    # dataset and depend on the result of /splits
    if name.startswith("empty-"):
        r_rows = poll(f"/first-rows?dataset={dataset}&config={config}&split={split}", error_field="error")
    elif name.startswith("missing-"):
        d = f"dataset={dataset}" if dataset is not None else ""
        c = f"config={config}" if config is not None else ""
        s = f"split={split}" if split is not None else ""
        params = "&".join([d, c, s])
        r_rows = poll(f"/first-rows?{params}", error_field="error")
    elif name.startswith("inexistent-") or name.startswith("private-") or name.startswith("gated-"):
        refresh_poll_splits(dataset)
        # no need to retry
        r_rows = get(f"/first-rows?dataset={dataset}&config={config}&split={split}")
    elif name == "not-ready":
        refresh_poll_splits(dataset)
        # poll the endpoint before the worker had the chance to process it
        r_rows = get(f"/first-rows?dataset={dataset}&config={config}&split={split}")
    else:
        _, r_rows = refresh_poll_splits_first_rows(dataset, config, split)

    assert r_rows.status_code == status, f"{r_rows.status_code} - {r_rows.text}"
    assert prepare_json(r_rows) == body, r_rows.text
    if error_code is not None:
        assert r_rows.headers["X-Error-Code"] == error_code, r_rows.headers["X-Error-Code"]
    else:
        assert "X-Error-Code" not in r_rows.headers, r_rows.headers["X-Error-Code"]


# from .utils import ROWS_MAX_NUMBER, URL, refresh_poll_splits_first_rows

# # TODO: find a dataset that can be processed faster
# def test_png_image():
#     # this test ensures that an image is saved as PNG if it cannot be saved as PNG
#     # https://github.com/huggingface/datasets-server/issues/191
#     dataset = "wikimedia/wit_base"
#     config = "wikimedia--wit_base"
#     split = "train"

#     _, r_rows = refresh_poll_splits_first_rows(dataset, config, split)

#     assert r_rows.status_code == 200, f"{r_rows.status_code} - {r_rows.text}"
#     json = r_rows.json()

#     assert "features" in json, json
#     assert json["features"][0]["name"] == "image", json
#     assert json["features"][0]["type"]["_type"] == "Image", json
#     assert (
#         json["rows"][0]["row"]["image"]
#         == f"{URL}/assets/wikimedia/wit_base/--/wikimedia--wit_base/train/0/image/image.jpg"
#     ), json

#     # assert (
#     #     json["rows"][20]["row"]["image"]
#     #     == f"{URL}/assets/wikimedia/wit_base/--/wikimedia--wit_base/train/20/image/image.png"
#     # )
#     # ^only four rows for now
