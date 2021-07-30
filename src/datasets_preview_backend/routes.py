from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.queries import extract_split_rows
from datasets_preview_backend.utils import get_int_value
from datasets_preview_backend.exceptions import (
    DatasetBuilderScriptError,
    DatasetBuilderScriptConfigNoSplitsError,
    DatasetNotFoundError,
    ConfigNotFoundError,
    SplitError,
    SplitNotImplementedError,
)


async def healthcheck(_: Request):
    return PlainTextResponse("ok")


async def rows(request: Request):
    dataset_id: str = request.query_params.get("dataset")
    config_name: str = request.query_params.get("config")
    split_name: str = request.query_params.get("split")
    num_rows = get_int_value(
        d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT
    )

    if dataset_id is None:
        return PlainTextResponse(
            "'dataset' is a required query parameter.", status_code=400
        )
    # note: config_name must not be set to refer to the None config_name (the majority of datasets).
    if split_name is None:
        return PlainTextResponse(
            "'split' is a required query parameter.", status_code=400
        )

    try:
        return JSONResponse(
            extract_split_rows(dataset_id, config_name, split_name, num_rows)
        )
    except (DatasetNotFoundError, ConfigNotFoundError) as err:
        return PlainTextResponse(err.message, status_code=404)
    except (
        DatasetBuilderScriptError,
        DatasetBuilderScriptConfigNoSplitsError,
        SplitError,
        SplitNotImplementedError,
    ) as err:
        return PlainTextResponse(err.message, status_code=400)
    # other exceptions will generate a 500 response
