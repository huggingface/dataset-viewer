from admin.scripts.refresh_cache_canonical import get_hf_canonical_dataset_names


# get_dataset_names
def test_get_hf_canonical_dataset_names() -> None:
    dataset_names = get_hf_canonical_dataset_names()
    assert len(dataset_names) > 100
    assert "glue" in dataset_names
    assert "Helsinki-NLP/tatoeba_mt" not in dataset_names
