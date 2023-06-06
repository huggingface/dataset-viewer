# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
import os

# see https://github.com/huggingface/moon-landing/blob/main/server/scripts/staging-seed-db.ts
CI_APP_TOKEN = "hf_app_datasets-server_token"
CI_PARQUET_CONVERTER_APP_TOKEN = "hf_app_datasets-server-parquet-converter_token"
CI_HUB_ENDPOINT = "https://hub-ci.huggingface.co"
CI_URL_TEMPLATE = CI_HUB_ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"
CI_USER = "__DUMMY_DATASETS_SERVER_USER__"
CI_USER_TOKEN = "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"
CI_SPAWNING_TOKEN = os.getenv("CI_SPAWNING_TOKEN", "unset")
