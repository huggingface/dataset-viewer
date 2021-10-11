from datasets_preview_backend.queries.cache_reports import get_cache_reports


def test_get_cache_stats() -> None:
    reports = get_cache_reports()["reports"]
    assert len(reports) > 100
    report = reports[0]
    assert "dataset" in report
    assert "status" in report
    assert "error" in report
