from dataclasses import asdict
from typing import Any, Dict, Optional

from datasets import load_dataset_builder

from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error

Info = Dict[str, Any]


def get_info(dataset_name: str, config_name: str, hf_token: Optional[str] = None) -> Info:
    try:
        # TODO: use get_dataset_infos if https://github.com/huggingface/datasets/issues/3013 is fixed
        builder = load_dataset_builder(
            dataset_name, name=config_name, download_mode=FORCE_REDOWNLOAD, use_auth_token=hf_token  # type: ignore
        )
        info = asdict(builder.info)
        if "splits" in info and info["splits"] is not None:
            info["splits"] = {split_name: split_info for split_name, split_info in info["splits"].items()}
    except Exception as err:
        raise Status400Error("Cannot get the metadata info for the config.", err)
    return info
