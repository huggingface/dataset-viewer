import pytest
from datasets import Dataset
from libutils.exceptions import CustomError

from worker.responses.first_rows import get_first_rows_response

from ..fixtures.files import DATA
from ..fixtures.hub import DatasetRepos, DatasetReposType
from ..utils import (
    ASSETS_BASE_URL,
    DEFAULT_HF_ENDPOINT,
    HF_ENDPOINT,
    HF_TOKEN,
    get_default_config_split,
)


@pytest.mark.parametrize(
    "type,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
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
    audio_dataset: Dataset,
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
    if type == "audio":
        column = "audio_column"
        assert response["features"][0]["name"] == column
        assert response["features"][0]["type"]["_type"] == "Audio"
        assert response["features"][0]["type"]["sampling_rate"] == audio_dataset.features[column].sampling_rate

        assert len(response["rows"]) == min(rows_max_number, len(audio_dataset))
        assert response["rows"][0]["row"] == {
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
        }
    else:
        assert response["features"][0]["name"] == "col_1"
        assert response["features"][0]["type"]["_type"] == "Value"
        assert response["features"][0]["type"]["dtype"] == "int64"  # <---|
        assert response["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
        assert response["features"][2]["type"]["dtype"] == "float64"  # <-|

        assert len(response["rows"]) == min(rows_max_number, len(DATA))
        assert response["rows"][0]["row_idx"] == 0
        assert response["rows"][0]["row"] == {"col_1": 0, "col_2": 0, "col_3": 0.0}
