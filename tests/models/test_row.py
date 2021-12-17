from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.models.config import get_config_names
from datasets_preview_backend.models.row import get_rows


# get_rows
def test_get_rows() -> None:
    rows = get_rows("acronym_identification", DEFAULT_CONFIG_NAME, "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert rows[0]["tokens"][0] == "What"


def test_class_label() -> None:
    rows = get_rows("glue", "cola", "train")
    assert rows[0]["label"] == 1


def test_mnist() -> None:
    rows = get_rows("mnist", "mnist", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert isinstance(rows[0]["image"], list)


def test_cifar() -> None:
    rows = get_rows("cifar10", "plain_text", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert isinstance(rows[0]["img"], list)


def test_iter_archive() -> None:
    rows = get_rows("food101", DEFAULT_CONFIG_NAME, "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert type(rows[0]["image"]["filename"]) == str
    assert type(rows[0]["image"]["data"]) == bytes


def test_dl_1_suffix() -> None:
    # see https://github.com/huggingface/datasets/pull/2843
    rows = get_rows("discovery", "discovery", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT


def test_txt_zip() -> None:
    # see https://github.com/huggingface/datasets/pull/2856
    rows = get_rows("bianet", "en_to_ku", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT


# TOO LONG... TODO: investigate why. Desactivating for now
# def test_pathlib() -> None:
#     # see https://github.com/huggingface/datasets/issues/2866
#     rows = get_rows("counter", DEFAULT_CONFIG_NAME, "train")
#     assert len(rows) == EXTRACT_ROWS_LIMIT


def test_community_with_no_config() -> None:
    config_names = get_config_names(dataset_name="Check/region_1")
    assert config_names == ["Check___region_1"]
    rows = get_rows("Check/region_1", "Check___region_1", "train")
    # it's not correct: here this is the number of splits, not the number of rows
    assert len(rows) == 2
    # see https://github.com/huggingface/datasets-preview-backend/issues/78
    get_rows("Check/region_1", "Check___region_1", "train")


def test_audio_dataset() -> None:
    rows = get_rows("common_voice", "tr", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert rows[0]["audio"]["sampling_rate"] == 48000
