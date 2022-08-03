from .utils import ROWS_MAX_NUMBER, URL, refresh_poll_splits_next_first_rows


def test_get_dataset_next():
    dataset = "acronym_identification"
    config = "default"
    split = "train"

    r_splits, r_rows = refresh_poll_splits_next_first_rows(dataset, config, split)
    assert r_splits.json()["splits"][0]["split_name"] == "train", f"{r_splits.status_code} - {r_splits.text}"

    assert r_rows.status_code == 200, f"{r_rows.status_code} - {r_rows.text}"
    json = r_rows.json()
    assert "features" in json, json
    assert json["features"][0]["name"] == "id", json
    assert json["features"][0]["type"]["_type"] == "Value", json
    assert json["features"][0]["type"]["dtype"] == "string", json
    assert json["features"][2]["name"] == "labels", json
    assert json["features"][2]["type"]["_type"] == "Sequence", json
    assert json["features"][2]["type"]["feature"]["_type"] == "ClassLabel", json
    assert json["features"][2]["type"]["feature"]["num_classes"] == 5, json
    assert "rows" in json
    assert len(json["rows"]) == ROWS_MAX_NUMBER, json["rows"]
    assert json["rows"][0]["row"]["id"] == "TR-0", json["rows"]
    assert type(json["rows"][0]["row"]["labels"]) is list, json["rows"]
    assert len(json["rows"][0]["row"]["labels"]) == 18, json["rows"]
    assert json["rows"][0]["row"]["labels"][0] == 4, json["rows"]


# TODO: find a dataset that can be processed faster
def test_png_image_next():
    # this test ensures that an image is saved as PNG if it cannot be saved as PNG
    # https://github.com/huggingface/datasets-server/issues/191
    dataset = "wikimedia/wit_base"
    config = "wikimedia--wit_base"
    split = "train"

    _, r_rows = refresh_poll_splits_next_first_rows(dataset, config, split)

    assert r_rows.status_code == 200, f"{r_rows.status_code} - {r_rows.text}"
    json = r_rows.json()

    assert "features" in json, json
    assert json["features"][0]["name"] == "image", json
    assert json["features"][0]["type"]["_type"] == "Image", json
    assert (
        json["rows"][0]["row"]["image"]
        == f"{URL}/assets/wikimedia/wit_base/--/wikimedia--wit_base/train/0/image/image.jpg",
        json,
    )
    # assert (
    #     json["rows"][20]["row"]["image"]
    #     == f"{URL}/assets/wikimedia/wit_base/--/wikimedia--wit_base/train/20/image/image.png"
    # )
    # ^only four rows for now
