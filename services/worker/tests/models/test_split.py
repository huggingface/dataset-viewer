import pandas
from worker.models.split import get_split

from .._utils import HF_TOKEN, ROWS_MAX_NUMBER

# TODO: test fallback


# TODO: this is slow: change the tested dataset?
def test_detect_types_from_typed_rows() -> None:
    split = get_split("allenai/c4", "allenai--c4", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert split["rows_response"]["columns"][0]["column"]["type"] == "STRING"


def test_class_label() -> None:
    split = get_split("glue", "cola", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert split["rows_response"]["rows"][0]["row"]["label"] == 1
    assert split["rows_response"]["columns"][1]["column"]["type"] == "CLASS_LABEL"
    assert "unacceptable" in split["rows_response"]["columns"][1]["column"]["labels"]


def test_mnist() -> None:
    split = get_split("mnist", "mnist", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert split["rows_response"]["rows"][0]["row"]["image"] == "assets/mnist/--/mnist/train/0/image/image.jpg"
    assert split["rows_response"]["columns"][0]["column"]["type"] == "RELATIVE_IMAGE_URL"


# TODO: re-enable the test
# def test_cifar() -> None:
#     info = get_info("cifar10", "plain_text")
#     typed_rows, columns = get_typed_rows_and_columns(
#         "cifar10", "plain_text", "train", info, rows_max_number=ROWS_MAX_NUMBER
#     )
#     assert len(typed_rows) == ROWS_MAX_NUMBER
#     assert typed_rows[0]["img"] == "assets/cifar10/--/plain_text/train/0/img/image.jpg"
#     assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


# TODO: re-enable the test
# def test_head_qa() -> None:
#     info = get_info("head_qa", "es")
#     typed_rows, columns = get_typed_rows_and_columns("head_qa", "es", "train", info, rows_max_number=ROWS_MAX_NUMBER)
#     assert len(typed_rows) == ROWS_MAX_NUMBER
#     assert typed_rows[0]["image"] is None
#     assert columns[6].name == "image"
#     assert columns[6].type == ColumnType.RELATIVE_IMAGE_URL


def test_iter_archive() -> None:
    split = get_split("food101", "default", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert split["rows_response"]["columns"][0]["column"]["type"] == "RELATIVE_IMAGE_URL"


def test_image_url() -> None:
    split = get_split("severo/wit", "default", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert split["rows_response"]["columns"][2]["column"]["type"] == "IMAGE_URL"


def test_audio_dataset() -> None:
    split = get_split("abidlabs/test-audio-1", "test", "train", rows_max_number=ROWS_MAX_NUMBER)
    assert len(split["rows_response"]["rows"]) == 1
    assert len(split["rows_response"]["rows"][0]["row"]["Output"]) == 2
    assert split["rows_response"]["rows"][0]["row"]["Output"][0]["type"] == "audio/mpeg"
    assert split["rows_response"]["rows"][0]["row"]["Output"][1]["type"] == "audio/wav"
    assert (
        split["rows_response"]["rows"][0]["row"]["Output"][0]["src"]
        == "assets/abidlabs/test-audio-1/--/test/train/0/Output/audio.mp3"
    )
    assert split["rows_response"]["columns"][1]["column"]["type"] == "AUDIO_RELATIVE_SOURCES"


def test_audio_path_none_dataset() -> None:
    split = get_split("LIUM/tedlium", "release1", "test", rows_max_number=ROWS_MAX_NUMBER)
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert len(split["rows_response"]["rows"][0]["row"]["audio"]) == 2
    assert split["rows_response"]["rows"][0]["row"]["audio"][0]["type"] == "audio/mpeg"
    assert split["rows_response"]["rows"][0]["row"]["audio"][1]["type"] == "audio/wav"
    assert (
        split["rows_response"]["rows"][0]["row"]["audio"][0]["src"]
        == "assets/LIUM/tedlium/--/release1/test/0/audio/audio.mp3"
    )
    assert split["rows_response"]["columns"][0]["column"]["type"] == "AUDIO_RELATIVE_SOURCES"


def test_get_split() -> None:
    dataset_name = "acronym_identification"
    config_name = "default"
    split_name = "train"
    split = get_split(dataset_name, config_name, split_name)

    assert split["num_bytes"] == 7792803
    assert split["num_examples"] == 14006


def test_gated() -> None:
    dataset_name = "severo/dummy_gated"
    config_name = "severo--embellishments"
    split_name = "train"
    split = get_split(dataset_name, config_name, split_name, HF_TOKEN, rows_max_number=ROWS_MAX_NUMBER)

    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert split["rows_response"]["rows"][0]["row"]["year"] == "1855"


def test_fallback() -> None:
    # https://github.com/huggingface/datasets/issues/3185
    dataset_name = "samsum"
    config_name = "samsum"
    split_name = "train"
    MAX_SIZE_FALLBACK = 100_000_000
    split = get_split(
        dataset_name,
        config_name,
        split_name,
        HF_TOKEN,
        rows_max_number=ROWS_MAX_NUMBER,
        max_size_fallback=MAX_SIZE_FALLBACK,
    )

    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER


def test_timestamp() -> None:

    ROWS_MAX_NUMBER = 1

    split = get_split(
        "ett",
        "h1",
        "train",
        HF_TOKEN,
        rows_max_number=ROWS_MAX_NUMBER,
    )
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    split["rows_response"]["rows"][0]
    split["rows_response"]["columns"][0]
    assert split["rows_response"]["rows"][0]["row"]["start"] == 1467331200.0
    assert split["rows_response"]["columns"][0]["column"]["type"] == "TIMESTAMP"
    assert split["rows_response"]["columns"][0]["column"]["unit"] == "s"
    assert split["rows_response"]["columns"][0]["column"]["tz"] is None
    # check
    assert (
        pandas.Timestamp(
            split["rows_response"]["rows"][0]["row"]["start"],
            unit=split["rows_response"]["columns"][0]["column"]["unit"],
            tz=split["rows_response"]["columns"][0]["column"]["tz"],
        ).isoformat()
        == "2016-07-01T00:00:00"
    )


# TODO: test the truncation
