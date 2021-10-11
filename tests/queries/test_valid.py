from datasets_preview_backend.queries.validity_status import get_valid_datasets


def test_get_valid_datasets() -> None:
    report = get_valid_datasets()
    assert "valid" in report
    assert "error" in report
    assert len(report["cache_miss"]) > 100
