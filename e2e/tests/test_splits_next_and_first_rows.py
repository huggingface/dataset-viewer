from .utils import (
    ROWS_MAX_NUMBER,
    URL,
    refresh_poll_splits_next_first_rows,
)


def test_get_dataset_next():
    dataset = "acronym_identification"
    config = "default"
    split = "train"

    r_splits, r_rows = refresh_poll_splits_next_first_rows(dataset, config, split)
    assert r_splits.json()["splits"][0]["split_name"] == "train"

    assert r_rows.status_code == 200
    json = r_rows.json()
    assert "features" in json
    assert json["features"][0]["name"] == "id"
    assert json["features"][0]["type"]["_type"] == "Value"
    assert json["features"][0]["type"]["dtype"] == "string"
    assert json["features"][2]["name"] == "labels"
    assert json["features"][2]["type"]["_type"] == "Sequence"
    assert json["features"][2]["type"]["feature"]["_type"] == "ClassLabel"
    assert json["features"][2]["type"]["feature"]["num_classes"] == 5
    assert "rows" in json
    assert len(json["rows"]) == ROWS_MAX_NUMBER
    assert json["rows"][0]["row"]["id"] == "TR-0"
    assert type(json["rows"][0]["row"]["labels"]) is list
    assert len(json["rows"][0]["row"]["labels"]) == 18
    assert json["rows"][0]["row"]["labels"][0] == 4


# TODO: find a dataset that can be processed faster
def test_png_image_next():
    # this test ensures that an image is saved as PNG if it cannot be saved as PNG
    # https://github.com/huggingface/datasets-server/issues/191
    dataset = "wikimedia/wit_base"
    config = "wikimedia--wit_base"
    split = "train"

    _, r_rows = refresh_poll_splits_next_first_rows(dataset, config, split)

    assert r_rows.status_code == 200
    json = r_rows.json()

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

