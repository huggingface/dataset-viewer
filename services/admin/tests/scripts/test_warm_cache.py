from admin.scripts.warm_cache import get_hf_dataset_names

from ..fixtures.hub import DatasetRepos


# get_dataset_names
def test_get_hf_dataset_names(hf_dataset_repos_csv_data: DatasetRepos) -> None:
    dataset_names = get_hf_dataset_names()
    assert len(dataset_names) >= 2
    assert hf_dataset_repos_csv_data["public"] in dataset_names
    assert hf_dataset_repos_csv_data["gated"] in dataset_names
    assert hf_dataset_repos_csv_data["private"] not in dataset_names
