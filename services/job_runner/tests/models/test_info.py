from job_runner.models.info import get_info


def test_get_info() -> None:
    info = get_info("glue", "ax")
    assert info.features is not None


def test_get_info_no_dataset_info_file() -> None:
    info = get_info("lhoestq/custom_squad", "plain_text")
    assert info.features is not None
