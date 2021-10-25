from datasets_preview_backend.io.mongo import get_client


def test_get_client() -> None:
    client = get_client()
    info = client.server_info()  # type: ignore
    assert info["ok"] == 1
