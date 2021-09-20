from dataclasses import asdict
from typing import Optional

from datasets import get_dataset_infos

from datasets_preview_backend.config import cache
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.responses import SerializedResponse
from datasets_preview_backend.types import InfoDict, ResponseJSON


def get_info(dataset: str, token: Optional[str] = None) -> InfoDict:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    try:
        total_dataset_infos = get_dataset_infos(dataset, use_auth_token=token)
        info = {config_name: asdict(config_info) for config_name, config_info in total_dataset_infos.items()}
    except FileNotFoundError as err:
        raise Status404Error("The dataset info could not be found.") from err
    except Exception as err:
        raise Status400Error("The dataset info could not be parsed from the dataset.") from err

    return {"dataset": dataset, "info": info}


@cache.memoize(expire=60)  # type:ignore
def get_info_json(dataset: str, token: Optional[str] = None) -> ResponseJSON:
    try:
        response = SerializedResponse(get_info(dataset, token))
    except (Status400Error, Status404Error) as err:
        response = SerializedResponse(err.as_dict(), err.status_code)
    return response.as_json()
