# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

# see https://github.com/huggingface/moon-landing/blob/main/server/scripts/staging-seed-db.ts
CI_APP_TOKEN = "hf_app_datasets-server_token"
CI_HUB_ENDPOINT = "https://hub-ci.huggingface.co"
PROD_HUB_ENDPOINT = "https://huggingface.co"
#
NORMAL_USER = "DVUser"
NORMAL_USER_TOKEN = "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"
NORMAL_ORG = "DVNormalOrg"
PRO_USER = "DVProUser"
PRO_USER_TOKEN = "hf_pro_user_token"
ENTERPRISE_ORG = "DVEnterpriseOrg"
ENTERPRISE_USER = "DVEnterpriseUser"
ENTERPRISE_USER_TOKEN = "hf_enterprise_user_token"

DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_REVISION = "some-revision"
ASSETS_BASE_URL = "https://baseurl/assets"
DEFAULT_SAMPLING_RATE = 16_000
DEFAULT_ROW_IDX = 0  # <- we cannot change this, because the fixtures only have one row
DEFAULT_COLUMN_NAME = "col"

DATASETS_NAMES = [
    "null",
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "time",
    "timestamp_1",
    "timestamp_2",
    "timestamp_3",
    "timestamp_tz",
    "string",
    "class_label",
    "dict",
    "list",
    "sequence_implicit",
    "sequence_list",
    "sequence_dict",
    "array2d",
    "array3d",
    "array4d",
    "array5d",
    "translation",
    "translation_variable_languages",
    "audio",
    "audio_ogg",
    "image",
    "images_list",
    "audios_list",
    "images_sequence",
    "images_sequence_dict",
    "audios_sequence",
    "dict_of_audios_and_images",
    "sequence_of_dicts",
    "none_value",
]

TEN_CHARS_TEXT = "a" * 10
DEFAULT_MIN_CELL_BYTES = 7
DEFAULT_ROWS_MAX_BYTES = 5_000
DEFAULT_CELL_BYTES = 91
DEFAULT_NUM_ROWS = 13
DEFAULT_NUM_CELLS = 1
DEFAULT_ROWS_MIN_NUMBER = 5
DEFAULT_ROWS_MAX_NUMBER = 73
DEFAULT_COLUMNS_MAX_NUMBER = 57
SOME_BYTES = 100
