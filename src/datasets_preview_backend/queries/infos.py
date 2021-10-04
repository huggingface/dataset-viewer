from dataclasses import asdict
from typing import List, Optional, cast

from datasets import load_dataset_builder

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, cache
from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.configs import get_configs_response
from datasets_preview_backend.responses import CachedResponse
from datasets_preview_backend.types import (
    ConfigsContent,
    InfoItem,
    InfosContent,
    StatusErrorContent,
)


def get_infos(dataset: str, config: Optional[str] = None) -> InfosContent:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    if dataset in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")
    if config is not None:
        if not isinstance(config, str):
            raise TypeError("config argument should be a string")
        configs = [config]
    else:
        content = get_config_responses(dataset=dataset).content
        if "configs" not in content:
            error = cast(StatusErrorContent, content)
            if "status_code" in error and error["status_code"] == 404:
                raise Status404Error("configurations could not be found")
            raise Status400Error("configurations could not be found")
        configs_content = cast(ConfigsContent, content)
        configs = [configItem["config"] for configItem in configs_content["configs"]]

    infoItems: List[InfoItem] = []
    # Note that we raise on the first error
    for config in configs:
        try:
            # TODO: use get_dataset_infos if https://github.com/huggingface/datasets/issues/3013 is fixed
            builder = load_dataset_builder(dataset, name=config)
            info = asdict(builder.info)
            if "splits" in info and info["splits"] is not None:
                info["splits"] = {split_name: split_info for split_name, split_info in info["splits"].items()}
        except FileNotFoundError as err:
            raise Status404Error("The config info could not be found.") from err
        except Exception as err:
            raise Status400Error("The config info could not be parsed from the dataset.") from err
        infoItems.append({"dataset": dataset, "config": config, "info": info})

    return {"infos": infoItems}


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_infos_response(*, dataset: str, config: Optional[str] = None) -> CachedResponse:
    try:
        response = CachedResponse(get_infos(dataset, config))
    except (Status400Error, Status404Error) as err:
        response = CachedResponse(err.as_content(), err.status_code)
    return response


def get_refreshed_infos(dataset: str, config: Optional[str] = None) -> InfosContent:
    return cast(InfosContent, get_infos_response(dataset, config, _refresh=True)["content"])
