import pytest

from datasets_preview_backend.queries.splits import (
    Status400Error,
    Status404Error,
    get_splits,
)


def test_get_splits():
    dataset = "acronym_identification"
    config = None
    response = get_splits(dataset, config)
    assert "dataset" in response
    assert response["dataset"] == dataset
    assert "config" in response
    assert response["config"] == config
    assert "splits" in response
    splits = response["splits"]
    assert len(splits) == 3
    assert "train" in splits

    splits = get_splits("glue", "ax")["splits"]
    assert len(splits) == 1
    assert "test" in splits
    assert "train" not in splits

    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    splits = get_splits("hda_nli_hindi", "HDA nli hindi")["splits"]
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    splits = get_splits("classla/copa_hr", "copa_hr")["splits"]
    assert len(splits) == 3

    splits = get_splits("mc4", "sn")["splits"]
    assert len(splits) == 2


def test_no_splits():
    # Due to https://github.com/huggingface/datasets/issues/2743
    with pytest.raises(Status400Error):
        get_splits("journalists_questions", "plain_text")


def test_builder_config_error():
    with pytest.raises(Status400Error):
        get_splits("KETI-AIR/nikl", "spoken.v1.0")
    with pytest.raises(Status400Error):
        get_splits("nateraw/image-folder", None)
    with pytest.raises(Status400Error):
        get_splits("Valahaar/wsdmt", None)


def test_not_found():
    with pytest.raises(Status404Error):
        get_splits("doesnotexist", None)
    with pytest.raises(Status404Error):
        get_splits("glue", "doesnotexist")
