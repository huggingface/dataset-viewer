import pytest
from datasets import Dataset

from libcommon.duckdb_utils import compute_length_column


def test_compute_length_column_for_list_of_json(tmp_path_factory: pytest.TempPathFactory) -> None:
    parquet_directory = tmp_path_factory.mktemp("data")
    parquet_filename = parquet_directory / "list_of_json.parquet"

    data = {"col": [[0, "a"]]}
    ds = Dataset.from_dict(data, on_mixed_types="use_json")
    ds.to_parquet(parquet_filename)

    length_column = compute_length_column([parquet_filename], "col", dtype="list", target_df=None)
    assert length_column.to_dict(as_series=False) == {"col.length": [2]}
