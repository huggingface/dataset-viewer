from PIL import Image  # type: ignore

from job_runner.models.row import get_rows

from .._utils import ROWS_MAX_NUMBER


# get_rows
def test_get_rows() -> None:
    rows = get_rows("acronym_identification", "default", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(rows) == ROWS_MAX_NUMBER
    assert rows[0]["tokens"][0] == "What"


def test_class_label() -> None:
    rows = get_rows("glue", "cola", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert rows[0]["label"] == 1


def test_mnist() -> None:
    rows = get_rows("mnist", "mnist", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(rows) == ROWS_MAX_NUMBER
    assert isinstance(rows[0]["image"], Image.Image)


def test_cifar() -> None:
    rows = get_rows("cifar10", "plain_text", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(rows) == ROWS_MAX_NUMBER
    assert isinstance(rows[0]["img"], Image.Image)


def test_iter_archive() -> None:
    rows = get_rows("food101", "default", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(rows) == ROWS_MAX_NUMBER
    assert isinstance(rows[0]["image"], Image.Image)


def test_dl_1_suffix() -> None:
    # see https://github.com/huggingface/datasets/pull/2843
    rows = get_rows("discovery", "discovery", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(rows) == ROWS_MAX_NUMBER


def test_txt_zip() -> None:
    # see https://github.com/huggingface/datasets/pull/2856
    rows = get_rows("bianet", "en_to_ku", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(rows) == ROWS_MAX_NUMBER


def test_pathlib() -> None:
    # see https://github.com/huggingface/datasets/issues/2866
    rows = get_rows("counter", "counter", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(rows) == ROWS_MAX_NUMBER


def test_community_with_no_config() -> None:
    rows = get_rows("Check/region_1", "Check--region_1", "train", rows_max_number=ROWS_MAX_NUMBER)
    # it's not correct: here this is the number of splits, not the number of rows
    assert len(rows) == 2
    # see https://github.com/huggingface/datasets-preview-backend/issues/78


def test_audio_dataset() -> None:
    rows = get_rows("abidlabs/test-audio-1", "test", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(rows) == 1
    assert rows[0]["Output"]["sampling_rate"] == 48000
