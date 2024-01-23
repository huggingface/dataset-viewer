# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

# see https://github.com/huggingface/moon-landing/blob/main/server/scripts/staging-seed-db.ts
CI_APP_TOKEN = "hf_app_datasets-server_token"
CI_HUB_ENDPOINT = "https://hub-ci.huggingface.co"
PROD_HUB_ENDPOINT = "https://huggingface.co"
#
NORMAL_USER = "DSSUser"
NORMAL_USER_TOKEN = "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"
NORMAL_ORG = "DSSNormalOrg"
PRO_USER = "DSSProUser"
PRO_USER_TOKEN = "hf_pro_user_token"
ENTERPRISE_ORG = "DSSEnterpriseOrg"
ENTERPRISE_USER = "DSSEnterpriseUser"
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
    "sequence_simple",
    "sequence",
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
    "audios_sequence",
    "dict_of_audios_and_images",
    "sequence_of_dicts",
    "none_value",
]
