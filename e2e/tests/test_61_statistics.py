# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from .fixtures.hub import AuthHeaders, AuthType
from .utils import get_default_config_split, poll_until_ready_and_assert


def test_statistics_endpoint(
    auth_headers: AuthHeaders,
    hf_public_dataset_repo_csv_data: str,
) -> None:
    auth: AuthType = "none"
    expected_status_code: int = 200
    expected_error_code = None
    # TODO: add dataset with various splits, or various configs
    dataset, config, split = get_default_config_split(hf_public_dataset_repo_csv_data)
    headers = auth_headers[auth]
    # ensure the /search endpoint works as well
    statistics_response = poll_until_ready_and_assert(
        relative_url=(
            f"/statistics?dataset={dataset}&config={config}&split={split}"
        ),
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=headers,
        check_x_revision=True,
    )
    if not expected_error_code:
        content = statistics_response.json()
        assert "num_examples" in content, statistics_response
        assert "statistics" in content, statistics_response

        statistics = content["statistics"]
        num_examples = content["num_examples"]
        assert isinstance(statistics, dict), statistics
        assert num_examples == 4
        assert statistics == {}