import pytest

from .utils import get_default_config_split, poll_until_ready_and_assert


def test_filter_endpoint(hf_public_dataset_repo_csv_data: str) -> None:
    # TODO: add dataset with various splits, or various configs
    dataset = hf_public_dataset_repo_csv_data
    config, split = get_default_config_split()
    offset = 1
    length = 2
    where = "col_4='B'"
    filter_response = poll_until_ready_and_assert(
        relative_url=(
            f"/filter?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}&where={where}"
        ),
        check_x_revision=True,
    )
    content = filter_response.json()
    assert "rows" in content, filter_response
    assert "features" in content, filter_response
    assert "num_rows_total" in content, filter_response
    assert "num_rows_per_page" in content, filter_response
    rows = content["rows"]
    features = content["features"]
    num_rows_total = content["num_rows_total"]
    num_rows_per_page = content["num_rows_per_page"]
    assert isinstance(rows, list), rows
    assert isinstance(features, list), features
    assert num_rows_total == 3
    assert num_rows_per_page == 100
    assert rows[0] == {
        "row_idx": 1,
        "row": {
            "col_1": "Vader turns round and round in circles as his ship spins into space.",
            "col_2": 1,
            "col_3": 1.0,
            "col_4": "B",
        },
        "truncated_cells": [],
    }, rows[0]
    assert rows[1] == {
        "row_idx": 3,
        "row": {
            "col_1": "The wingman spots the pirateship coming at him and warns the Dark Lord",
            "col_2": 3,
            "col_3": 3.0,
            "col_4": "B",
        },
        "truncated_cells": [],
    }, rows[1]
    assert features == [
        {"feature_idx": 0, "name": "col_1", "type": {"dtype": "string", "_type": "Value"}},
        {"feature_idx": 1, "name": "col_2", "type": {"dtype": "int64", "_type": "Value"}},
        {"feature_idx": 2, "name": "col_3", "type": {"dtype": "float64", "_type": "Value"}},
        {"feature_idx": 3, "name": "col_4", "type": {"dtype": "string", "_type": "Value"}},
    ], features


@pytest.mark.parametrize(
    "where, expected_num_rows",
    [
        ("col_2=3", 1),
        ("col_2<3", 3),
        ("col_2>3", 0),
        ("col_4='B'", 3),
        ("col_4<'B'", 1),
        ("col_4>='A'", 4),
        ("col_2<3 AND col_4='B'", 2),
        ("col_2<3 OR col_4='B'", 4),
    ],
)
def test_filter_endpoint_parameter_where(
    where: str, expected_num_rows: int, hf_public_dataset_repo_csv_data: str
) -> None:
    dataset = hf_public_dataset_repo_csv_data
    config, split = get_default_config_split()
    response = poll_until_ready_and_assert(
        relative_url=f"/filter?dataset={dataset}&config={config}&split={split}&where={where}",
        check_x_revision=True,
    )
    content = response.json()
    assert "rows" in content, response
    assert len(content["rows"]) == expected_num_rows
