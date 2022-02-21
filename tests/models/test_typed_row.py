from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.models.column import ClassLabelColumn, ColumnType
from datasets_preview_backend.models.info import get_info
from datasets_preview_backend.models.typed_row import get_typed_rows_and_columns

# def test_detect_types_from_typed_rows() -> None:
#     info = get_info("allenai/c4", "")
#     typed_rows, columns = get_typed_rows_and_columns("allenai/c4", "default", "train", info)
#     assert len(typed_rows) == EXTRACT_ROWS_LIMIT
#     assert columns[0].type == ColumnType.STRING


def test_class_label() -> None:
    info = get_info("glue", "cola")
    typed_rows, columns = get_typed_rows_and_columns("glue", "cola", "train", info)
    column = columns[1]
    assert isinstance(column, ClassLabelColumn)
    assert column.type == ColumnType.CLASS_LABEL
    assert "unacceptable" in column.labels
    assert typed_rows[0]["label"] == 1


def test_mnist() -> None:
    info = get_info("mnist", "mnist")
    typed_rows, columns = get_typed_rows_and_columns("mnist", "mnist", "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert typed_rows[0]["image"] == "assets/mnist/--/mnist/train/0/image/image.jpg"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_cifar() -> None:
    info = get_info("cifar10", "plain_text")
    typed_rows, columns = get_typed_rows_and_columns("cifar10", "plain_text", "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert typed_rows[0]["img"] == "assets/cifar10/--/plain_text/train/0/img/image.jpg"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


# disable until https://github.com/huggingface/datasets/issues/3758 is fixed
# def test_head_qa() -> None:
#     info = get_info("head_qa", "es")
#     typed_rows, columns = get_typed_rows_and_columns("head_qa", "es", "train", info)
#     assert len(typed_rows) == EXTRACT_ROWS_LIMIT
#     assert typed_rows[0]["image"] is None
#     assert columns[6].name == "image"
#     assert columns[6].type == ColumnType.RELATIVE_IMAGE_URL


def test_iter_archive() -> None:
    info = get_info("food101", "default")
    typed_rows, columns = get_typed_rows_and_columns("food101", "default", "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_image_url() -> None:
    info = get_info("severo/wit", "default")
    typed_rows, columns = get_typed_rows_and_columns("severo/wit", "default", "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert columns[2].type == ColumnType.IMAGE_URL


def test_audio_dataset() -> None:
    info = get_info("common_voice", "tr")
    typed_rows, columns = get_typed_rows_and_columns("common_voice", "tr", "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert columns[2].type == ColumnType.AUDIO_RELATIVE_SOURCES
    assert len(typed_rows[0]["audio"]) == 2
    assert typed_rows[0]["audio"][0]["type"] == "audio/mpeg"
    assert typed_rows[0]["audio"][1]["type"] == "audio/wav"
    assert typed_rows[0]["audio"][0]["src"] == "assets/common_voice/--/tr/train/0/audio/audio.mp3"
