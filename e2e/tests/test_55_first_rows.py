# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from .utils import get_default_config_split, poll, poll_until_ready_and_assert


def test_first_rows_images_endpoint(normal_user_images_public_dataset: str) -> None:
    dataset = normal_user_images_public_dataset
    config, split = get_default_config_split()
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/first-rows?dataset={dataset}&config={config}&split={split}",
        dataset=dataset,
        should_retry_x_error_codes=["ResponseNotFound"],
        # ^ I had 404 errors without it. It should return something else at one point.
    )
    content = rows_response.json()
    # ensure the URL is signed
    url = content["rows"][0]["row"]["image"]["src"]
    assert isinstance(url, str), url
    assert "image.jpg?Expires=" in url, url
    assert "&Signature=" in url, url
    assert "&Key-Pair-Id=" in url, url
    # ensure the URL has been signed only once
    assert url.count("Expires") == 1, url
    # ensure the URL is valid
    response = poll(url, url="")
    assert response.status_code == 200, response


def test_first_rows_audios_endpoint(normal_user_audios_public_dataset: str) -> None:
    dataset = normal_user_audios_public_dataset
    config, split = get_default_config_split()
    rows_response = poll_until_ready_and_assert(
        relative_url=f"/first-rows?dataset={dataset}&config={config}&split={split}",
        dataset=dataset,
        should_retry_x_error_codes=["ResponseNotFound"],
        # ^ I had 404 errors without it. It should return something else at one point.
    )
    content = rows_response.json()
    # ensure the URL is signed
    url = content["rows"][0]["row"]["audio"][0]["src"]
    assert isinstance(url, str), url
    assert "audio.wav?Expires=" in url, url
    assert "&Signature=" in url, url
    assert "&Key-Pair-Id=" in url, url
    # ensure the URL has been signed only once
    assert url.count("Expires") == 1, url
    # ensure the URL is valid
    response = poll(url, url="")
    assert response.status_code == 200, response
