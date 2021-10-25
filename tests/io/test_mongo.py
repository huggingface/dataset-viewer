from datasets_preview_backend.io.mongo import connect_cache


def test_connect_cache() -> None:
    connect_cache()
