from dataclasses import asdict
from typing import Any, Dict

from datasets import load_dataset_builder

from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error, Status404Error

Info = Dict[str, Any]


def get_info(dataset_name: str, config_name: str) -> Info:
    try:
        # TODO: use get_dataset_infos if https://github.com/huggingface/datasets/issues/3013 is fixed
        builder = load_dataset_builder(dataset_name, name=config_name, download_mode=FORCE_REDOWNLOAD)  # type: ignore
        info = asdict(builder.info)
        if "splits" in info and info["splits"] is not None:
            info["splits"] = {split_name: split_info for split_name, split_info in info["splits"].items()}
    except FileNotFoundError as err:
        raise Status404Error("The config info could not be found.", err)
    except Exception as err:
        raise Status400Error("The config info could not be parsed from the dataset.", err)
    return info
