import pytest

from datasets_preview_backend.queries.splits import (
    DatasetBuilderNoSplitsError,
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

    # uses the fallback to call "builder._split_generators"
    splits = get_splits("hda_nli_hindi", "HDA nli hindi")["splits"]
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    splits = get_splits("classla/copa_hr", "copa_hr")["splits"]
    assert len(splits) == 3

    splits = get_splits("mc4", "sn")["splits"]
    assert len(splits) == 2


def test_extract_bogus_splits():
    # not sure if we have an example of such an error
    with pytest.raises(DatasetBuilderNoSplitsError):
        get_splits("journalists_questions", "plain_text")
