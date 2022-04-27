import base64
import sys
from enum import Enum, auto
from typing import Any, Dict, List, TypedDict

import orjson
from pymongo import MongoClient

from libcache.cache import Status
from libcache.config import MONGO_CACHE_DATABASE, MONGO_URL

client = MongoClient(MONGO_URL)
db = client[MONGO_CACHE_DATABASE]


# copy code required for the migration (it might disappear in next iterations)
class RowItem(TypedDict):
    dataset: str
    config: str
    split: str
    row_idx: int
    row: Dict[str, Any]
    truncated_cells: List[str]


class ColumnType(Enum):
    JSON = auto()  # default
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    IMAGE_URL = auto()
    RELATIVE_IMAGE_URL = auto()
    AUDIO_RELATIVE_SOURCES = auto()
    CLASS_LABEL = auto()


def get_empty_rows_response() -> Dict[str, Any]:
    return {"columns": [], "rows": []}


def to_column_item(column: Dict[str, Any]) -> Dict[str, Any]:
    column_field = {
        "name": column["name"],
        "type": ColumnType(column["type"]).name,
    }
    if "labels" in column and len(column["labels"]) > 0:
        column_field["labels"] = column["labels"]

    return {
        "dataset": column["dataset_name"],
        "config": column["config_name"],
        "split": column["split_name"],
        "column_idx": column["column_idx"],
        "column": column_field,
    }


# orjson is used to get rid of errors with datetime (see allenai/c4)
def orjson_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    raise TypeError


def orjson_dumps(content: Any) -> bytes:
    return orjson.dumps(content, option=orjson.OPT_UTC_Z, default=orjson_default)


def get_size_in_bytes(obj: Any):
    return sys.getsizeof(orjson_dumps(obj))
    # ^^ every row is transformed here in a string, because it corresponds to
    # the size the row will contribute in the JSON response to /rows endpoint.
    # The size of the string is measured in bytes.
    # An alternative would have been to look at the memory consumption (pympler) but it's
    # less related to what matters here (size of the JSON, number of characters in the
    # dataset viewer table on the hub)


def truncate_cell(cell: Any, min_cell_bytes: int) -> str:
    return orjson_dumps(cell)[:min_cell_bytes].decode("utf8", "ignore")


DEFAULT_MIN_CELL_BYTES = 100


# Mutates row_item, and returns it anyway
def truncate_row_item(row_item: RowItem) -> RowItem:
    min_cell_bytes = DEFAULT_MIN_CELL_BYTES
    row = {}
    for column_name, cell in row_item["row"].items():
        # for now: all the cells, but the smallest ones, are truncated
        cell_bytes = get_size_in_bytes(cell)
        if cell_bytes > min_cell_bytes:
            row_item["truncated_cells"].append(column_name)
            row[column_name] = truncate_cell(cell, min_cell_bytes)
        else:
            row[column_name] = cell
    row_item["row"] = row
    return row_item


# Mutates row_items, and returns them anyway
def truncate_row_items(row_items: List[RowItem], rows_max_bytes: int) -> List[RowItem]:
    # compute the current size
    rows_bytes = sum(get_size_in_bytes(row_item) for row_item in row_items)

    # Loop backwards, so that the last rows are truncated first
    for row_item in reversed(row_items):
        if rows_bytes < rows_max_bytes:
            break
        previous_size = get_size_in_bytes(row_item)
        row_item = truncate_row_item(row_item)
        new_size = get_size_in_bytes(row_item)
        rows_bytes += new_size - previous_size
    return row_items


def to_row_item(row: Dict[str, Any]) -> RowItem:
    return {
        "dataset": row["dataset_name"],
        "config": row["config_name"],
        "split": row["split_name"],
        "row_idx": row["row_idx"],
        "row": row["row"],
        "truncated_cells": [],
    }


# migrate
rows_max_bytes = 1_000_000
splits_coll = db.splits
rows_coll = db.rows
columns_coll = db.columns
splits_coll.update_many({}, {"$set": {"rows_response": get_empty_rows_response()}})
# ^ add the new field to all the splits
for split in splits_coll.find({"status": {"$in": [Status.VALID.value, Status.STALLED.value]}}):
    print(f"update split {split}")
    columns = list(
        columns_coll.find(
            {
                "dataset_name": split["dataset_name"],
                "config_name": split["config_name"],
                "split_name": split["split_name"],
            }
        )
    )
    print(f"found {len(columns)} columns")
    rows = list(
        rows_coll.find(
            {
                "dataset_name": split["dataset_name"],
                "config_name": split["config_name"],
                "split_name": split["split_name"],
            }
        )
    )
    print(f"found {len(rows)} rows")
    column_items = [to_column_item(column) for column in sorted(columns, key=lambda d: d["column_idx"])]
    row_items = truncate_row_items(
        [to_row_item(row) for row in sorted(rows, key=lambda d: d["row_idx"])], rows_max_bytes
    )
    rows_response = {"columns": column_items, "rows": row_items}
    splits_coll.update_one({"_id": split["_id"]}, {"$set": {"rows_response": rows_response}})

# ^ fill the rows_response field, only for VALID and STALLED
db["rows"].drop()
db["columns"].drop()
