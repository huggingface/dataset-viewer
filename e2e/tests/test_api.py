import json
import os
import time
from typing import Optional

import pytest
import requests

SERVICE_REVERSE_PROXY_PORT = os.environ.get("SERVICE_REVERSE_PROXY_PORT", "8000")
INTERVAL = 1
MAX_DURATION = 10 * 60

URL = f"http://localhost:{SERVICE_REVERSE_PROXY_PORT}"


def poll_until_valid_response(url: str, error_field: Optional[str] = None) -> requests.Response:
    interval = INTERVAL
    timeout = MAX_DURATION
    retries = timeout // interval
    should_retry = True
    response = None
    while retries > 0 and should_retry:
        retries -= 1
        time.sleep(interval)
        response = requests.get(url)
        if error_field is not None:
            # currently, when the dataset is being processed, the error message contains "Retry later"
            try:
                should_retry = "retry later" in response.json()[error_field].lower()
            except Exception:
                should_retry = False
        else:
            # just retry if the response is not 200
            should_retry = response.status_code != 200
    if response is None:
        raise RuntimeError("no request has been done")
    return response


def poll_splits_until_dataset_process_has_finished(dataset: str, endpoint: str = "splits") -> requests.Response:
    error_field = "message" if endpoint == "splits" else "error"
    return poll_until_valid_response(f"{URL}/{endpoint}?dataset={dataset}", error_field=error_field)


def poll_rows_until_split_process_has_finished(
    dataset: str, config: str, split: str, endpoint: str = "rows"
) -> requests.Response:
    error_field = "message" if endpoint == "rows" else "error"
    return poll_until_valid_response(
        f"{URL}/{endpoint}?dataset={dataset}&config={config}&split={split}", error_field=error_field
    )


def test_healthcheck():
    # this tests ensures the nginx reverse proxy and the api are up
    response = poll_until_valid_response(f"{URL}/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


def test_valid():
    # this test ensures that the mongo db can be accessed by the api
    response = poll_until_valid_response(f"{URL}/valid")
    assert response.status_code == 200
    # at this moment no dataset has been processed
    assert response.json()["valid"] == []


@pytest.mark.usefixtures("config")
def test_splits_next_200(config):
    with open(config["openapi"]) as json_file:
        openapi = json.load(json_file)
    expected_status = 200
    expected_body = openapi["paths"]["/splits-next"]["get"]["responses"][str(expected_status)]["content"][
        "application/json"
    ]["examples"]["duorc"]["value"]

    # ask for the dataset to be refreshed
    dataset = "duorc"
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    # poll the /splits endpoint until we get something else than "The dataset is being processed. Retry later."
    response = poll_splits_until_dataset_process_has_finished(dataset, "splits-next")
    assert response.status_code == 200

    response = requests.get(f"{URL}/splits-next?dataset={dataset}")
    assert response.status_code == expected_status
    assert response.json() == expected_body


@pytest.mark.usefixtures("config")
def test_splits_next_404(config):
    with open(config["openapi"]) as json_file:
        openapi = json.load(json_file)
    expected_status = 404
    expected_body = openapi["paths"]["/splits-next"]["get"]["responses"][str(expected_status)]["content"][
        "application/json"
    ]["examples"]["inexistent-dataset"]["value"]
    response = requests.get(f"{URL}/splits-next?dataset=inexistent_dataset")
    assert response.status_code == expected_status
    assert response.json() == expected_body


def test_get_dataset():
    dataset = "acronym_identification"
    config = "default"
    split = "train"

    # ask for the dataset to be refreshed
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    # poll the /splits endpoint until we get something else than "The dataset is being processed. Retry later."
    response = poll_splits_until_dataset_process_has_finished(dataset, "splits")
    assert response.status_code == 200

    # poll the /rows endpoint until we get something else than "The split is being processed. Retry later."
    response = poll_rows_until_split_process_has_finished(dataset, config, split, "rows")
    assert response.status_code == 200
    json = response.json()
    assert "rows" in json
    assert json["rows"][0]["row"]["id"] == "TR-0"


def test_get_dataset_next():
    dataset = "acronym_identification"
    config = "default"
    split = "train"

    # ask for the dataset to be refreshed
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    # poll the /splits endpoint until we get something else than "The dataset is being processed. Retry later."
    response = poll_splits_until_dataset_process_has_finished(dataset, "splits-next")
    assert response.status_code == 200

    # poll the /rows endpoint until we get something else than "The split is being processed. Retry later."
    response = poll_rows_until_split_process_has_finished(dataset, config, split, "first-rows")
    assert response.status_code == 200
    json = response.json()

    assert "features" in json
    assert json["features"][0]["name"] == "id"
    assert json["features"][0]["type"]["_type"] == "Value"
    assert json["features"][0]["type"]["dtype"] == "string"
    assert json["features"][2]["name"] == "labels"
    assert json["features"][2]["type"]["_type"] == "Sequence"
    assert json["features"][2]["type"]["feature"]["_type"] == "ClassLabel"
    assert json["features"][2]["type"]["feature"]["num_classes"] == 5
    assert "rows" in json
    assert json["rows"][0]["row"]["id"] == "TR-0"
    assert type(json["rows"][0]["row"]["labels"]) is list
    assert len(json["rows"][0]["row"]["labels"]) == 18
    assert json["rows"][0]["row"]["labels"][0] == 4


def test_bug_empty_split():
    # see #185 and #177
    # we get an error when:
    # - the dataset has been processed and the splits have been created in the database
    # - the splits have not been processed and are still in EMPTY status in the database
    # - the dataset is processed again, and the splits are marked as STALE
    # - they are thus returned with an empty content, instead of an error message
    # (waiting for being processsed)
    dataset = "nielsr/CelebA-faces"
    config = "nielsr--CelebA-faces"
    split = "train"

    # ask for the dataset to be refreshed
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    # poll the /splits endpoint until we get something else than "The dataset is being processed. Retry later."
    response = poll_splits_until_dataset_process_has_finished(dataset, "splits")
    assert response.status_code == 200

    # at this point the splits should have been created in the dataset, and still be EMPTY
    url = f"{URL}/rows?dataset={dataset}&config={config}&split={split}"
    response = requests.get(url)
    assert response.status_code == 400
    json = response.json()
    assert json["message"] == "The split is being processed. Retry later."

    # ask again for the dataset to be refreshed
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    # at this moment, there is a concurrency race between the datasets worker and the splits worker
    # but the dataset worker should finish before, because it's faster on this dataset
    # With the bug, if we polled again /rows until we have something else than "being processed",
    # we would have gotten a valid response, but with empty rows, which is incorrect
    # Now: it gives a correct list of elements
    response = poll_rows_until_split_process_has_finished(dataset, config, split, "rows")
    assert response.status_code == 200
    json = response.json()
    assert len(json["rows"]) == 100


def test_valid_after_datasets_processed():
    # this test ensures that the datasets processed successfully are present in /valid
    response = requests.get(f"{URL}/valid")
    assert response.status_code == 200
    # at this moment various datasets have been processed
    assert "acronym_identification" in response.json()["valid"]
    assert "nielsr/CelebA-faces" in response.json()["valid"]


# TODO: enable this test (not sure why it fails)
# def test_timestamp_column():
#     # this test replicates the bug with the Timestamp values, https://github.com/huggingface/datasets/issues/4413
#     dataset = "ett"
#     config = "h1"
#     split = "train"
#     response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
#     assert response.status_code == 200

#     response = poll_splits_until_dataset_process_has_finished(dataset, "splits", 60)
#     assert response.status_code == 200

#     response = poll_rows_until_split_process_has_finished(dataset, config, split, "rows", 60)
#     assert response.status_code == 200
#     json = response.json()
#     TRUNCATED_TO_ONE_ROW = 1
#     assert len(json["rows"]) == TRUNCATED_TO_ONE_ROW
#     assert json["rows"][0]["row"]["start"] == 1467331200.0
#     assert json["columns"][0]["column"]["type"] == "TIMESTAMP"
#     assert json["columns"][0]["column"]["unit"] == "s"
#     assert json["columns"][0]["column"]["tz"] is None


def test_png_image():
    # this test ensures that an image is saved as PNG if it cannot be saved as PNG
    # https://github.com/huggingface/datasets-server/issues/191
    dataset = "wikimedia/wit_base"
    config = "wikimedia--wit_base"
    split = "train"
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    response = poll_splits_until_dataset_process_has_finished(dataset, "splits")
    assert response.status_code == 200

    response = poll_rows_until_split_process_has_finished(dataset, config, split, "rows")
    assert response.status_code == 200
    json = response.json()
    assert json["columns"][0]["column"]["type"] == "RELATIVE_IMAGE_URL"
    assert (
        json["rows"][0]["row"]["image"] == "assets/wikimedia/wit_base/--/wikimedia--wit_base/train/0/image/image.jpg"
    )
    assert (
        json["rows"][20]["row"]["image"] == "assets/wikimedia/wit_base/--/wikimedia--wit_base/train/20/image/image.png"
    )


def test_png_image_next():
    # this test ensures that an image is saved as PNG if it cannot be saved as PNG
    # https://github.com/huggingface/datasets-server/issues/191
    dataset = "wikimedia/wit_base"
    config = "wikimedia--wit_base"
    split = "train"
    response = requests.post(f"{URL}/webhook", json={"update": f"datasets/{dataset}"})
    assert response.status_code == 200

    response = poll_splits_until_dataset_process_has_finished(dataset, "splits-next")
    assert response.status_code == 200

    response = poll_rows_until_split_process_has_finished(dataset, config, split, "first-rows")
    assert response.status_code == 200
    json = response.json()

    assert "features" in json
    assert json["features"][0]["name"] == "image"
    assert json["features"][0]["type"]["_type"] == "Image"
    assert (
        json["rows"][0]["row"]["image"]
        == f"{URL}/assets/wikimedia/wit_base/--/wikimedia--wit_base/train/0/image/image.jpg"
    )
    assert (
        json["rows"][20]["row"]["image"]
        == f"{URL}/assets/wikimedia/wit_base/--/wikimedia--wit_base/train/20/image/image.png"
    )
