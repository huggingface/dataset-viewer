from datasets_preview_backend.io.mongo import (
    get_client,
    get_database,
    get_datasets_collection,
)


def test_get_client() -> None:
    client = get_client()
    info = client.server_info()  # type: ignore
    assert info["ok"] == 1


def test_get_database() -> None:
    get_database()


def test_get_datasets_collection() -> None:
    get_datasets_collection()
