from typing import Dict

import pytest
from datasets import Dataset
from libutils.exceptions import CustomError

from worker.responses.first_rows import get_first_rows_response

from ..fixtures.hub import DatasetRepos, DatasetReposType
from ..utils import ASSETS_BASE_URL, HF_ENDPOINT, HF_TOKEN, get_default_config_split


@pytest.mark.parametrize(
    "type,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("image", False, None, None),
        # TODO: re-enable both when https://github.com/huggingface/datasets/issues/4875 is fixed
        # ("gated", True, None, None),
        # ("private", True, None, None),  # <- TODO: should we disable accessing private datasets?
        ("empty", False, "SplitsNamesError", "FileNotFoundError"),
        ("does_not_exist", False, "DatasetNotFoundError", None),
        ("gated", False, "SplitsNamesError", "FileNotFoundError"),
        ("private", False, "SplitsNamesError", "FileNotFoundError"),
    ],
)
def test_number_rows(
    hf_dataset_repos_csv_data: DatasetRepos,
    type: DatasetReposType,
    use_token: bool,
    error_code: str,
    cause: str,
    datasets: Dict[str, Dataset],
) -> None:
    rows_max_number = 7
    dataset, config, split = get_default_config_split(hf_dataset_repos_csv_data[type])
    if error_code is not None:
        with pytest.raises(CustomError) as exc_info:
            get_first_rows_response(
                dataset_name=dataset,
                config_name=config,
                split_name=split,
                assets_base_url=ASSETS_BASE_URL,
                hf_endpoint=HF_ENDPOINT,
                hf_token=HF_TOKEN if use_token else None,
                rows_max_number=rows_max_number,
            )
        assert exc_info.value.code == error_code
        if cause is None:
            assert exc_info.value.disclose_cause is False
            assert exc_info.value.cause_exception is None
        else:
            assert exc_info.value.disclose_cause is True
            assert exc_info.value.cause_exception == cause
            response = exc_info.value.as_response()
            assert set(response.keys()) == {"error", "cause_exception", "cause_message", "cause_traceback"}
            assert response["error"] == "Cannot get the split names for the dataset."
            response_dict = dict(response)
            # ^ to remove mypy warnings
            assert response_dict["cause_exception"] == "FileNotFoundError"
            assert str(response_dict["cause_message"]).startswith("Couldn't find a dataset script at ")
            assert isinstance(response_dict["cause_traceback"], list)
            assert response_dict["cause_traceback"][0] == "Traceback (most recent call last):\n"
        return
    response = get_first_rows_response(
        dataset_name=dataset,
        config_name=config,
        split_name=split,
        assets_base_url=ASSETS_BASE_URL,
        hf_endpoint=HF_ENDPOINT,
        hf_token=HF_TOKEN if use_token else None,
        rows_max_number=rows_max_number,
    )
    assert response["features"][0]["feature_idx"] == 0
    assert response["rows"][0]["row_idx"] == 0
    column = "col"
    if type == "audio":
        assert response == {
            "features": [
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "feature_idx": 0,
                    "name": column,
                    "type": {
                        "_type": "Audio",
                        "decode": True,
                        "id": None,
                        "mono": True,
                        "sampling_rate": datasets["audio"].features[column].sampling_rate,
                    },
                }
            ],
            "rows": [
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "row_idx": 0,
                    "truncated_cells": [],
                    "row": {
                        column: [
                            {
                                "src": f"http://localhost/assets/{dataset}/--/{config}/{split}/0/{column}/audio.mp3",
                                "type": "audio/mpeg",
                            },
                            {
                                "src": f"http://localhost/assets/{dataset}/--/{config}/{split}/0/{column}/audio.wav",
                                "type": "audio/wav",
                            },
                        ]
                    },
                }
            ],
        }
    elif type == "image":
        assert response == {
            "features": [
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "feature_idx": 0,
                    "name": column,
                    "type": {
                        "_type": "Image",
                        "decode": True,
                        "id": None,
                    },
                }
            ],
            "rows": [
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "row_idx": 0,
                    "truncated_cells": [],
                    "row": {
                        column: f"http://localhost/assets/{dataset}/--/{config}/{split}/0/{column}/image.jpg",
                    },
                }
            ],
        }
    else:
        assert response == {
            "features": [
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "feature_idx": 0,
                    "name": "col_1",
                    "type": {"_type": "Value", "id": None, "dtype": "int64"},
                },
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "feature_idx": 1,
                    "name": "col_2",
                    "type": {"_type": "Value", "id": None, "dtype": "int64"},
                },
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "feature_idx": 2,
                    "name": "col_3",
                    "type": {"_type": "Value", "id": None, "dtype": "float64"},
                },
            ],
            "rows": [
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "row_idx": 0,
                    "truncated_cells": [],
                    "row": {"col_1": 0, "col_2": 0, "col_3": 0.0},
                },
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "row_idx": 1,
                    "truncated_cells": [],
                    "row": {"col_1": 1, "col_2": 1, "col_3": 1.0},
                },
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "row_idx": 2,
                    "truncated_cells": [],
                    "row": {"col_1": 2, "col_2": 2, "col_3": 2.0},
                },
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "row_idx": 3,
                    "truncated_cells": [],
                    "row": {"col_1": 3, "col_2": 3, "col_3": 3.0},
                },
            ],
        }
