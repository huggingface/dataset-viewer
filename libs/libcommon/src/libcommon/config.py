# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

from environs import Env


class CommonConfig:
    assets_base_url: str
    hf_endpoint: str
    hf_token: Optional[str]
    log_level: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("COMMON_"):
            self.assets_base_url = env.str(name="ASSETS_BASE_URL", default="assets")
            self.hf_endpoint = env.str(name="HF_ENDPOINT", default="https://huggingface.co")
            self.hf_token = env.str(name="HF_TOKEN", default=None)
            self.log_level = env.log_level(name="LOG_LEVEL", default="INFO")
