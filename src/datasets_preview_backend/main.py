import logging
import os

from datasets.builder import DatasetBuilder
from typing import List

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse
from starlette.routing import Route
import uvicorn

from datasets import load_dataset, prepare_module, import_main_class

DEFAULT_PORT = 8000
DEFAULT_EXTRACT_ROWS_LIMIT = 100


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class ConfigNameError(Error):
    """Exception raised for errors in the config name.

    Attributes:
        config_name -- the erroneous dataset config_name
    """

    def __init__(self, config_name):
        self.config_name = config_name


def get_int_value(d, key, default):
    try:
        value = int(d.get(key))
    except TypeError:
        value = default
    return value


PORT = get_int_value(d=os.environ, key="DPB_PORT", default=DEFAULT_PORT)
EXTRACT_ROWS_LIMIT = get_int_value(
    d=os.environ, key="DPB_EXTRACT_ROWS_LIMIT", default=DEFAULT_EXTRACT_ROWS_LIMIT
)


async def healthcheck(request: Request):
    return PlainTextResponse("ok")


def get_dataset_config_names(dataset_id: str) -> List[str]:
    module_path, *_ = prepare_module(dataset_id, dataset=True)
    builder_cls = import_main_class(module_path, dataset=True)
    config_names = [c.name for c in builder_cls.BUILDER_CONFIGS] or [None]
    logging.debug(
        f"The dataset builder has {len(config_names)} configs: {config_names}"
    )
    return config_names


def get_dataset_config_extract(dataset_id: str, config_name: str, num_rows: int):
    try:
        dataset = load_dataset(
            dataset_id, name=config_name, split="train", streaming=True
        )
    except ValueError as err:
        message = str(err)
        if message.startswith(f"BuilderConfig {config_name} not found"):
            raise ConfigNameError(config_name=config_name)
        else:
            raise

    logging.debug(f"Dataset loaded")

    rows = list(dataset.take(num_rows))

    if len(rows) != num_rows:
        logging.warning(
            f"could not read all the required rows ({len(rows)} / {num_rows})"
        )

    return {"dataset_id": dataset_id, "config_name": config_name, "rows": rows}


def get_dataset_extract(dataset_id: str, num_rows: int):
    # TODO: manage splits
    logging.debug(f"Asked for {num_rows} first rows of dataset {dataset_id}")

    config_names = get_dataset_config_names(dataset_id)

    return {
        "dataset_id": dataset_id,
        "configs": {
            config_name: get_dataset_config_extract(dataset_id, config_name, num_rows)
            for config_name in config_names
        },
    }


async def extract(request: Request):
    dataset_id: str = request.path_params["dataset_id"]
    num_rows = get_int_value(
        d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT
    )

    try:
        return JSONResponse(get_dataset_extract(dataset_id, num_rows))
    except FileNotFoundError as e:
        return PlainTextResponse("Dataset not found", status_code=404)
    # other exceptions will generate a 500 response


def start():
    app = Starlette(
        routes=[
            Route("/healthcheck", endpoint=healthcheck),
            Route("/{dataset_id:path}/extract", endpoint=extract),
        ]
    )

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    start()
