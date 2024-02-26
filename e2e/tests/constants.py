# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

# see https://github.com/huggingface/moon-landing/blob/main/server/scripts/staging-seed-db.ts
CI_APP_TOKEN = "hf_app_datasets-server_token"
CI_HUB_ENDPOINT = "https://hub-ci.huggingface.co"
NORMAL_USER = "DSSUser"
NORMAL_USER_TOKEN = "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"
NORMAL_USER_COOKIE = "oMidckPVQYumfKrAHNYKqnbacRoLaMppHRRlfNbupNahzAHCzInBVbhgGosDneYXHVTKkkWygoMDxBfFUkFPIPiVWBtZtSTYIYTScnEKAJYkyGBAcbVTbokAygCCTWvH"
NORMAL_ORG = "DSSNormalOrg"
PRO_USER = "DSSProUser"
PRO_USER_TOKEN = "hf_pro_user_token"
ENTERPRISE_ORG = "DSSEnterpriseOrg"
ENTERPRISE_USER = "DSSEnterpriseUser"
ENTERPRISE_USER_TOKEN = "hf_enterprise_user_token"

DATA = [
    {
        "col_1": "There goes another one.",
        "col_2": 0,
        "col_3": 0.0,
        "col_4": "B",
        "col_5": True,
        "col_6": [0, 1, 2],
    },
    {
        "col_1": "Vader turns round and round in circles as his ship spins into space.",
        "col_2": 1,
        "col_3": 1.0,
        "col_4": "B",
        "col_5": False,
        "col_6": [0, 1, 2, 3],
    },
    {
        "col_1": "We count thirty Rebel ships, Lord Vader.",
        "col_2": 2,
        "col_3": 2.0,
        "col_4": "A",
        "col_5": True,
        "col_6": [0, 1, 2, 3, None],
    },
    {
        "col_1": "The wingman spots the pirateship coming at him and warns the Dark Lord",
        "col_2": 3,
        "col_3": 3.0,
        "col_4": "B",
        "col_5": None,
        "col_6": None,
    },
]
