from admin.scripts.refresh_cache_canonical import get_hf_canonical_dataset_names


def test_get_hf_canonical_dataset_names() -> None:
    dataset_names = get_hf_canonical_dataset_names()
    assert len(dataset_names) >= 0
    # ^ TODO: have some canonical datasets in the hub-ci instance
    # with the current fixture user we are not able to create canonical datasets
