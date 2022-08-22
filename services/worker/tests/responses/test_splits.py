import pytest
from libutils.exceptions import CustomError

from worker.responses.splits import get_dataset_split_full_names, get_splits_response

from ..fixtures.hub import DatasetRepos, DatasetReposType
from ..utils import HF_ENDPOINT, HF_TOKEN, get_default_config_split


# TODO: remove this test? It's an internal function after all
@pytest.mark.parametrize(
    "type,use_token,exception",
    [
        ("empty", False, FileNotFoundError),
        ("does_not_exist", False, FileNotFoundError),
        ("public", False, None),
        ("gated", False, FileNotFoundError),
        ("gated", True, None),
        ("private", False, FileNotFoundError),
        ("private", True, None),  # <- TODO: should we disable accessing private datasets?
    ],
)
def test_get_dataset_split_full_names_simple_csv(
    hf_dataset_repos_csv_data: DatasetRepos, type: DatasetReposType, use_token: bool, exception: BaseException
) -> None:
    if exception:
        with pytest.raises(FileNotFoundError):
            get_dataset_split_full_names(
                dataset_name=hf_dataset_repos_csv_data[type], hf_token=HF_TOKEN if use_token else None
            )
        return
    split_full_names = get_dataset_split_full_names(
        dataset_name=hf_dataset_repos_csv_data[type], hf_token=HF_TOKEN if use_token else None
    )
    dataset, config, split = get_default_config_split(hf_dataset_repos_csv_data[type])
    assert split_full_names == [{"dataset_name": dataset, "config_name": config, "split_name": split}]


@pytest.mark.parametrize(
    "type,use_token,error_code,cause",
    [
        ("empty", False, "SplitsNamesError", "FileNotFoundError"),
        ("does_not_exist", False, "DatasetNotFoundError", None),
        ("public", False, None, None),
        ("gated", False, "SplitsNamesError", "FileNotFoundError"),
        ("gated", True, None, None),
        ("private", False, "SplitsNamesError", "FileNotFoundError"),
        ("private", True, None, None),  # <- TODO: should we disable accessing private datasets?
    ],
)
def test_get_splits_response_simple_csv(
    hf_dataset_repos_csv_data: DatasetRepos, type: DatasetReposType, use_token: bool, error_code: str, cause: str
) -> None:
    if error_code:
        with pytest.raises(CustomError) as exc_info:
            get_splits_response(hf_dataset_repos_csv_data[type], HF_ENDPOINT, HF_TOKEN if use_token else None)
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
    splits_response = get_splits_response(
        hf_dataset_repos_csv_data[type], HF_ENDPOINT, HF_TOKEN if use_token else None
    )
    dataset, config, split = get_default_config_split(hf_dataset_repos_csv_data[type])
    assert splits_response == {
        "splits": [
            {
                "dataset_name": dataset,
                "config_name": config,
                "split_name": split,
                "num_bytes": None,
                "num_examples": None,
            }
        ]
    }


# @pytest.mark.real_dataset
# def test_script_error() -> None:
#     # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
#     # which should be caught and raised as DatasetBuilderScriptError
#     with pytest.raises(ModuleNotFoundError):
#         get_dataset_split_full_names(dataset_name="piEsposito/br-quad-2.0")


# @pytest.mark.real_dataset
# def test_builder_config_error() -> None:
#     with pytest.raises(SplitsNotFoundError):
#         get_dataset_split_full_names(dataset_name="KETI-AIR/nikl")
#     with pytest.raises(RuntimeError):
#         get_dataset_split_full_names(dataset_name="nateraw/image-folder")
#     with pytest.raises(TypeError):
#         get_dataset_split_full_names(dataset_name="Valahaar/wsdmt")
