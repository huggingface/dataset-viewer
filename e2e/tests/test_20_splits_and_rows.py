from .fixtures.hub import DatasetRepos
from .utils import (
    ROWS_MAX_NUMBER,
    get,
    poll_rows,
    poll_splits,
    post,
    post_refresh,
    get_default_config_split,
    refresh_poll_splits_rows,
)


# TODO: find a dataset that can be processed faster
def test_bug_empty_split(hf_dataset_repos_csv_data: DatasetRepos):
    # see #185 and #177
    # we get an error when:
    # - the dataset has been processed and the splits have been created in the database
    # - the splits have not been processed and are still in EMPTY status in the database
    # - the dataset is processed again, and the splits are marked as STALE
    # - they are thus returned with an empty content, instead of an error message
    # (waiting for being processsed)

    dataset, config, split = get_default_config_split(hf_dataset_repos_csv_data["public2"])

    # ask for the dataset to be refreshed
    response = post_refresh(dataset)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"

    # poll the /splits endpoint until we get something else than "The dataset is being processed. Retry later."
    response = poll_splits(dataset)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"

    # at this point the splits should have been created in the dataset, and still be EMPTY
    response = get(f"/rows?dataset={dataset}&config={config}&split={split}")
    assert response.status_code == 400, f"{response.status_code} - {response.text}"
    json = response.json()
    assert json["message"] == "The split is being processed. Retry later.", json

    # ask again for the dataset to be refreshed
    response = post("/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200, f"{response.status_code} - {response.text}"

    # at this moment, there is a concurrency race between the datasets worker and the splits worker
    # but the dataset worker should finish before, because it's faster on this dataset
    # With the bug, if we polled again /rows until we have something else than "being processed",
    # we would have gotten a valid response, but with empty rows, which is incorrect
    # Now: it gives a correct list of elements
    response = poll_rows(dataset, config, split)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    json = response.json()
    assert len(json["rows"]) == ROWS_MAX_NUMBER, json


def test_get_dataset(hf_dataset_repos_csv_data: DatasetRepos):
    dataset, config, split = get_default_config_split(hf_dataset_repos_csv_data["public2"])

    r_splits, r_rows = refresh_poll_splits_rows(dataset, config, split)
    assert r_splits.json()["splits"][0]["split"] == "train", r_splits.text
    assert r_rows.json()["rows"][0]["row"]["col_1"] == 0, r_splits.text


# TODO: enable again when we will have the same behavior with 4 rows (ROWS_MAX_NUMBER)
# TODO: find a dataset that can be processed faster
# def test_png_image():
#     # this test ensures that an image is saved as PNG if it cannot be saved as PNG
#     # https://github.com/huggingface/datasets-server/issues/191
#     dataset = "wikimedia/wit_base"
#     config = "wikimedia--wit_base"
#     split = "train"

#     _, r_rows = refresh_poll_splits_rows(dataset, config, split)

#     json = r_rows.json()
#     assert json["columns"][0]["column"]["type"] == "RELATIVE_IMAGE_URL"
#     assert (
#         json["rows"][0]["row"]["image"] == "assets/wikimedia/wit_base/--/wikimedia--wit_base/train/0/image/image.jpg"
#     )
#     assert (
#         json["rows"][20]["row"]["image"] ==
#               "assets/wikimedia/wit_base/--/wikimedia--wit_base/train/20/image/image.png"
#     )


# TODO: enable this test (not sure why it fails)
# def test_timestamp_column():
#     # this test replicates the bug with the Timestamp values, https://github.com/huggingface/datasets/issues/4413
#     dataset = "ett"
#     config = "h1"
#     split = "train"
#     _, r_rows = refresh_poll_splits_rows(dataset, config, split)
#     json = r_rows.json()
#     TRUNCATED_TO_ONE_ROW = 1
#     assert len(json["rows"]) == TRUNCATED_TO_ONE_ROW
#     assert json["rows"][0]["row"]["start"] == 1467331200.0
#     assert json["columns"][0]["column"]["type"] == "TIMESTAMP"
#     assert json["columns"][0]["column"]["unit"] == "s"
#     assert json["columns"][0]["column"]["tz"] is None
