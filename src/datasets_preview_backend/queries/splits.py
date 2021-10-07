from typing import List, Optional

from datasets import get_dataset_split_names

from datasets_preview_backend.cache import cache, memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS
from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.types import SplitItem, SplitsContent


@memoize(cache=cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_splits(*, dataset: str, config: Optional[str] = None) -> SplitsContent:
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
        # note: the function might raise
        configs_content = get_configs(dataset=dataset)
        configs = [configItem["config"] for configItem in configs_content["configs"]]

    splitItems: List[SplitItem] = []
    # Note that we raise on the first error
    for config in configs:
        try:
            splits = get_dataset_split_names(dataset, config)
        except FileNotFoundError as err:
            raise Status404Error("The dataset config could not be found.", err)
        except ValueError as err:
            if str(err).startswith(f"BuilderConfig {config} not found."):
                raise Status404Error("The dataset config could not be found.", err)
            else:
                raise Status400Error("The split names could not be parsed from the dataset config.", err)
        except Exception as err:
            raise Status400Error("The split names could not be parsed from the dataset config.", err)
        splitItems += [{"dataset": dataset, "config": config, "split": split} for split in splits]

    return {"splits": splitItems}
