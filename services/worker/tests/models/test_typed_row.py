from worker.models.column import ClassLabelColumn, ColumnType
from worker.models.info import get_info
from worker.models.typed_row import get_typed_rows_and_columns

from .._utils import ROWS_MAX_NUMBER


# TODO: this is slow: change the tested dataset?
def test_detect_types_from_typed_rows() -> None:
    info = get_info("allenai/c4", "allenai--c4")
    typed_rows, columns = get_typed_rows_and_columns(
        "allenai/c4", "allenai--c4", "train", info, rows_max_number=ROWS_MAX_NUMBER
    )
    assert len(typed_rows) == ROWS_MAX_NUMBER
    assert columns[0].type == ColumnType.STRING


def test_class_label() -> None:
    info = get_info("glue", "cola")
    typed_rows, columns = get_typed_rows_and_columns("glue", "cola", "train", info, rows_max_number=ROWS_MAX_NUMBER)
    column = columns[1]
    assert isinstance(column, ClassLabelColumn)
    assert column.type == ColumnType.CLASS_LABEL
    assert "unacceptable" in column.labels
    assert typed_rows[0]["label"] == 1


def test_mnist() -> None:
    info = get_info("mnist", "mnist")
    typed_rows, columns = get_typed_rows_and_columns("mnist", "mnist", "train", info, rows_max_number=ROWS_MAX_NUMBER)
    assert len(typed_rows) == ROWS_MAX_NUMBER
    assert typed_rows[0]["image"] == "assets/mnist/--/mnist/train/0/image/image.jpg"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


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
    info = get_info("food101", "default")
    typed_rows, columns = get_typed_rows_and_columns(
        "food101", "default", "train", info, rows_max_number=ROWS_MAX_NUMBER
    )
    assert len(typed_rows) == ROWS_MAX_NUMBER
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_image_url() -> None:
    info = get_info("severo/wit", "default")
    typed_rows, columns = get_typed_rows_and_columns(
        "severo/wit", "default", "train", info, rows_max_number=ROWS_MAX_NUMBER
    )
    assert len(typed_rows) == ROWS_MAX_NUMBER
    assert columns[2].type == ColumnType.IMAGE_URL


def test_audio_dataset() -> None:
    info = get_info("abidlabs/test-audio-1", "test")
    typed_rows, columns = get_typed_rows_and_columns(
        "abidlabs/test-audio-1", "test", "train", info, rows_max_number=ROWS_MAX_NUMBER
    )
    assert len(typed_rows) == 1
    assert columns[1].type == ColumnType.AUDIO_RELATIVE_SOURCES
    assert len(typed_rows[0]["Output"]) == 2
    assert typed_rows[0]["Output"][0]["type"] == "audio/mpeg"
    assert typed_rows[0]["Output"][1]["type"] == "audio/wav"
    assert typed_rows[0]["Output"][0]["src"] == "assets/abidlabs/test-audio-1/--/test/train/0/Output/audio.mp3"
