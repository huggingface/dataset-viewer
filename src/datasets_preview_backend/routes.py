from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.queries import extract_dataset_rows
from datasets_preview_backend.utils import get_int_value
from datasets_preview_backend.exceptions import (
    DatasetBuilderScriptError,
    DatasetBuilderScriptConfigError,
    DatasetBuilderScriptConfigNoSplitsError,
    DatasetNotFoundError,
    ConfigNotFoundError,
    SplitError,
    SplitNotImplementedError,
)


async def healthcheck(_: Request):
    return PlainTextResponse("ok")


async def extract(request: Request):
    dataset_id: str = request.path_params["dataset_id"]
    num_rows = get_int_value(
        d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT
    )

    try:
        return JSONResponse(extract_dataset_rows(dataset_id, num_rows))
    except (DatasetNotFoundError, ConfigNotFoundError) as err:
        return PlainTextResponse(err.message, status_code=404)
    except (
        DatasetBuilderScriptError,
        DatasetBuilderScriptConfigError,
        DatasetBuilderScriptConfigNoSplitsError,
        SplitError,
        SplitNotImplementedError,
    ) as err:
        return PlainTextResponse(err.message, status_code=400)
    # other exceptions will generate a 500 response
