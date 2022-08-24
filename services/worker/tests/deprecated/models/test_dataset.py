import pytest
from datasets.inspect import SplitsNotFoundError

from worker.deprecated.models.dataset import get_dataset_split_full_names

# from ...utils import HF_TOKEN
pytestmark = pytest.mark.deprecated


def test_script_error() -> None:
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(ModuleNotFoundError):
        get_dataset_split_full_names(dataset_name="piEsposito/br-quad-2.0")


def test_no_dataset() -> None:
    # the dataset does not exist
    with pytest.raises(FileNotFoundError):
        get_dataset_split_full_names(dataset_name="doesnotexist")


def test_no_dataset_no_script() -> None:
    # the dataset does not contain a script
    with pytest.raises(FileNotFoundError):
        get_dataset_split_full_names(dataset_name="AConsApart/anime_subtitles_DialoGPT")
    with pytest.raises(FileNotFoundError):
        get_dataset_split_full_names(dataset_name="TimTreasure4/Test")


def test_builder_config_error() -> None:
    with pytest.raises(SplitsNotFoundError):
        get_dataset_split_full_names(dataset_name="KETI-AIR/nikl")
    with pytest.raises(RuntimeError):
        get_dataset_split_full_names(dataset_name="nateraw/image-folder")
    with pytest.raises(TypeError):
        get_dataset_split_full_names(dataset_name="Valahaar/wsdmt")


# get_split
def test_get_split() -> None:
    split_full_names = get_dataset_split_full_names("glue")
    assert len(split_full_names) == 34
    assert {"dataset_name": "glue", "config_name": "ax", "split_name": "test"} in split_full_names


def test_splits_fallback() -> None:
    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    split_full_names = get_dataset_split_full_names("hda_nli_hindi")
    assert len(split_full_names) == 3
    assert {"dataset_name": "hda_nli_hindi", "config_name": "HDA nli hindi", "split_name": "train"} in split_full_names


# disable until https://github.com/huggingface/datasets-server/pull/499 is done
# def test_gated() -> None:
#     split_full_names = get_dataset_split_full_names("severo/dummy_gated", HF_TOKEN)
#     assert len(split_full_names) == 1
#     assert {
#         "dataset_name": "severo/dummy_gated",
#         "config_name": "severo--embellishments",
#         "split_name": "train",
#     } in split_full_names
