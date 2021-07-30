import logging
import os

from datasets.builder import DatasetBuilder
from typing import List

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse
from starlette.routing import Route
import uvicorn

from datasets import (
    IterableDataset,
    load_dataset,
    load_dataset_builder,
    prepare_module,
    import_main_class,
)

from datasets_preview_backend.exceptions import (
    DatasetBuilderScriptError,
    DatasetBuilderScriptConfigError,
    DatasetNotFoundError,
    ConfigNotFoundError,
    SplitError,
)

DEFAULT_PORT = 8000
DEFAULT_EXTRACT_ROWS_LIMIT = 100


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
    try:
        module_path, *_ = prepare_module(dataset_id, dataset=True)
        builder_cls = import_main_class(module_path, dataset=True)
    except FileNotFoundError as err:
        raise DatasetNotFoundError(dataset_id=dataset_id)
    except (ModuleNotFoundError):
        raise DatasetBuilderScriptError(dataset_id=dataset_id)

    config_names = [c.name for c in builder_cls.BUILDER_CONFIGS] or [None]
    logging.debug(
        f"The dataset builder has {len(config_names)} configs: {config_names}"
    )
    return config_names


def get_config_splits(dataset_id: str, config_name: str) -> List[str]:
    try:
        builder = load_dataset_builder(dataset_id, name=config_name)
    except ValueError as err:
        message = str(err)
        if message.startswith(f"BuilderConfig {config_name} not found"):
            raise ConfigNotFoundError(dataset_id=dataset_id, config_name=config_name)
        else:
            raise
    except ModuleNotFoundError:
        raise DatasetBuilderScriptConfigError(
            dataset_id=dataset_id, config_name=config_name
        )

    if builder.info.splits is None:
        raise DatasetBuilderScriptConfigError(
            dataset_id=dataset_id, config_name=config_name
        )
    return builder.info.splits.keys()


def extract_split_rows(dataset_id: str, config_name: str, split: str, num_rows: int):
    logging.debug(
        f"asked for {num_rows} first rows of dataset {dataset_id} - {config_name} - {split}"
    )

    try:
        dataset: IterableDataset = load_dataset(
            dataset_id, name=config_name, split=split, streaming=True
        )
    except ValueError as err:
        message = str(err)
        if message.startswith(f"BuilderConfig {config_name} not found"):
            raise ConfigNotFoundError(dataset_id=dataset_id, config_name=config_name)
        elif message.startswith(f'Unknown split "{split}".') or message.startswith(
            f"Bad split: {split}."
        ):
            raise SplitError(
                dataset_id=dataset_id, config_name=config_name, split=split
            )
        else:
            raise

    rows = list(dataset.take(num_rows))
    if len(rows) != num_rows:
        logging.warning(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset_id} - {config_name} - {split}"
        )

    return {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "rows": rows,
    }


def extract_config_rows(dataset_id: str, config_name: str, num_rows: int):
    logging.debug(
        f"asked for {num_rows} first rows of dataset {dataset_id} - {config_name}"
    )

    splits = get_config_splits(dataset_id, config_name)

    return {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "splits": {
            split: extract_split_rows(dataset_id, config_name, split, num_rows)
            for split in splits
        },
    }


def extract_dataset_rows(dataset_id: str, num_rows: int):
    logging.debug(f"asked for {num_rows} first rows of dataset {dataset_id}")

    config_names = get_dataset_config_names(dataset_id)

    return {
        "dataset_id": dataset_id,
        "configs": {
            config_name: extract_config_rows(dataset_id, config_name, num_rows)
            for config_name in config_names
        },
    }


async def extract(request: Request):
    dataset_id: str = request.path_params["dataset_id"]
    num_rows = get_int_value(
        d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT
    )

    try:
        return JSONResponse(extract_dataset_rows(dataset_id, num_rows))
    except (DatasetNotFoundError, ConfigNotFoundError) as err:
        return PlainTextResponse(err.message, status_code=404)
    except (DatasetBuilderScriptError, DatasetBuilderScriptConfigError) as err:
        return PlainTextResponse(err.message, status_code=400)
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
