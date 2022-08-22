import pytest

from worker.responses.first_rows import get_first_rows_response

from ..utils import ASSETS_BASE_URL, HF_ENDPOINT


@pytest.mark.real_dataset
def test_number_rows() -> None:
    rows_max_number = 7
    response = get_first_rows_response(
        "duorc",
        "SelfRC",
        "train",
        rows_max_number=rows_max_number,
        assets_base_url=ASSETS_BASE_URL,
        hf_endpoint=HF_ENDPOINT,
    )
    assert len(response["rows"]) == rows_max_number


@pytest.mark.real_dataset
def test_get_first_rows_response() -> None:
    rows_max_number = 7
    response = get_first_rows_response(
        "common_voice",
        "tr",
        "train",
        rows_max_number=rows_max_number,
        assets_base_url=ASSETS_BASE_URL,
        hf_endpoint=HF_ENDPOINT,
    )

    assert response["features"][0]["feature_idx"] == 0
    assert response["features"][0]["name"] == "client_id"
    assert response["features"][0]["type"]["_type"] == "Value"
    assert response["features"][0]["type"]["dtype"] == "string"

    assert response["features"][2]["name"] == "audio"
    assert response["features"][2]["type"]["_type"] == "Audio"
    assert response["features"][2]["type"]["sampling_rate"] == 48000

    assert len(response["rows"]) == rows_max_number
    assert response["rows"][0]["row_idx"] == 0
    assert response["rows"][0]["row"]["client_id"].startswith("54fc2d015c27a057b")
    assert response["rows"][0]["row"]["audio"] == [
        {"src": f"{ASSETS_BASE_URL}/common_voice/--/tr/train/0/audio/audio.mp3", "type": "audio/mpeg"},
        {"src": f"{ASSETS_BASE_URL}/common_voice/--/tr/train/0/audio/audio.wav", "type": "audio/wav"},
    ]


@pytest.mark.real_dataset
def test_no_features() -> None:
    response = get_first_rows_response(
        "severo/fix-401",
        "severo--fix-401",
        "train",
        rows_max_number=1,
        assets_base_url=ASSETS_BASE_URL,
        hf_endpoint=HF_ENDPOINT,
    )

    # TODO: re-enable when we understand why it works locally but not in the CI (order of the features)
    # assert response["features"][5]["feature_idx"] == 5
    # assert response["features"][5]["name"] == "area_mean"
    # assert response["features"][5]["type"]["_type"] == "Value"
    # assert response["features"][5]["type"]["dtype"] == "float64"

    assert response["rows"][0]["row_idx"] == 0
    assert response["rows"][0]["row"]["diagnosis"] == "M"
    assert response["rows"][0]["row"]["area_mean"] == 1001.0
