import logging
from mimetypes import guess_type
from typing import Any, Optional, TypedDict

from datasets_preview_backend.dataset_entries import (
    filter_config_entries,
    filter_split_entries,
    get_dataset_entry,
)
from datasets_preview_backend.exceptions import Status400Error, Status404Error

logger = logging.getLogger(__name__)


class ImageCell(TypedDict):
    filename: str
    data: bytes
    mime_type: str


def get_image(
    dataset: Optional[str] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
    row: Optional[str] = None,
    column: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageCell:
    if dataset is None or config is None or split is None or row is None or column is None or filename is None:
        raise Status400Error("Some required parameters are missing (dataset, config, split, row, column or filename)")
    try:
        row_index = int(row)
    except (TypeError, ValueError):
        raise Status400Error("row must be an integer")

    dataset_entry = get_dataset_entry(dataset=dataset)

    config_entries = filter_config_entries(dataset_entry["configs"], config)
    try:
        config_entry = config_entries[0]
    except Exception as err:
        raise Status404Error("config not found", err)

    split_entries = filter_split_entries(config_entry["splits"], split)
    try:
        rows = split_entries[0]["rows"]
    except Exception as err:
        raise Status404Error("split not found", err)

    try:
        rowEntry = rows[row_index]
    except Exception as err:
        raise Status404Error("row not found", err)

    try:
        colEntry: Any = rowEntry[column]
    except Exception as err:
        raise Status404Error("column not found", err)

    try:
        if type(colEntry["filename"]) != str:
            raise TypeError("'filename' field must be a string")
    except Exception as err:
        raise Status400Error("cell has no filename field", err)

    if colEntry["filename"] != filename:
        raise Status404Error("filename not found in cell")

    try:
        data = colEntry["data"]
        if type(data) != bytes:
            raise TypeError("'data' field must be a bytes")
    except Exception as err:
        raise Status400Error("cell has no data field", err)

    mime_type = guess_type(filename)[0] or "text/plain"

    return {"filename": filename, "data": data, "mime_type": mime_type}
