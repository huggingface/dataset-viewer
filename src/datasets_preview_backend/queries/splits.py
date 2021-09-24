from typing import List, Optional, cast

from datasets import get_dataset_split_names

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, cache
from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.configs import get_configs_response
from datasets_preview_backend.responses import CachedResponse
from datasets_preview_backend.types import ConfigsContent, SplitItem, SplitsContent


def get_splits(dataset: str, config: Optional[str] = None, token: Optional[str] = None) -> SplitsContent:
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
        configs_content = cast(ConfigsContent, get_configs_response(dataset=dataset, token=token).content)
        if "configs" not in configs_content:
            raise Status400Error("configurations could not be found")
        configs = [configItem["config"] for configItem in configs_content["configs"]]

    splitItems: List[SplitItem] = []
    # Note that we raise on the first error
    for config in configs:
        try:
            splits = get_dataset_split_names(dataset, config, use_auth_token=token)
        except FileNotFoundError as err:
            raise Status404Error("The dataset config could not be found.") from err
        except ValueError as err:
            if str(err).startswith(f"BuilderConfig {config} not found."):
                raise Status404Error("The dataset config could not be found.") from err
            else:
                raise Status400Error("The split names could not be parsed from the dataset config.") from err
        except Exception as err:
            raise Status400Error("The split names could not be parsed from the dataset config.") from err
        splitItems += [{"dataset": dataset, "config": config, "split": split} for split in splits]

    return {"splits": splitItems}


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_splits_response(*, dataset: str, config: Optional[str] = None, token: Optional[str] = None) -> CachedResponse:
    try:
        response = CachedResponse(get_splits(dataset, config, token))
    except (Status400Error, Status404Error) as err:
        response = CachedResponse(err.as_content(), err.status_code)
    return response


def get_refreshed_splits(dataset: str, config: Optional[str] = None, token: Optional[str] = None) -> SplitsContent:
    return cast(SplitsContent, get_splits_response(dataset, config, token, _refresh=True)["content"])
