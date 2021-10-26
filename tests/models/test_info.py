from datasets_preview_backend.models.info import get_info


def test_get_info() -> None:
    info = get_info("glue", "ax")
    assert "features" in info


def test_get_info_no_dataset_info_file() -> None:
    info = get_info("lhoestq/custom_squad", "plain_text")
    assert "features" in info
