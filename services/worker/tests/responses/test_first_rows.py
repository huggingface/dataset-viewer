from worker.responses.first_rows import get_first_rows_response

from .._utils import ASSETS_BASE_URL


def test_get_first_rows_response() -> None:
    response = get_first_rows_response(
        "common_voice", "tr", "train", rows_max_number=1, assets_base_url=ASSETS_BASE_URL
    )

    assert response["features"][0]["feature_idx"] == 0
    assert response["features"][0]["name"] == "client_id"
    assert response["features"][0]["type"]["_type"] == "Value"
    assert response["features"][0]["type"]["dtype"] == "string"

    assert response["features"][2]["name"] == "audio"
    assert response["features"][2]["type"]["_type"] == "Audio"
    assert response["features"][2]["type"]["sampling_rate"] == 48000

    assert response["rows"][0]["row_idx"] == 0
    assert response["rows"][0]["row"]["client_id"].startswith("54fc2d015c27a057b")
    assert response["rows"][0]["row"]["audio"] == [
        {"src": f"{ASSETS_BASE_URL}/common_voice/--/tr/train/0/audio/audio.mp3", "type": "audio/mpeg"},
        {"src": f"{ASSETS_BASE_URL}/common_voice/--/tr/train/0/audio/audio.wav", "type": "audio/wav"},
    ]


def test_no_features() -> None:
    response = get_first_rows_response(
        "severo/fix-401", "severo--fix-401", "train", rows_max_number=1, assets_base_url=ASSETS_BASE_URL
    )

    assert response["features"][1]["feature_idx"] == 1
    assert response["features"][1]["name"] == "area_mean"
    assert response["features"][1]["type"]["_type"] == "Value"
    assert response["features"][1]["type"]["dtype"] == "float64"

    assert response["rows"][0]["row_idx"] == 0
    assert response["rows"][0]["row"]["area_mean"] == 1001.0
