from datasets_preview_backend.queries.cache_stats import get_cache_stats


def test_get_cache_stats() -> None:
    datasets = get_cache_stats()
    assert datasets["expected"] > 100
    assert "valid" in datasets
    assert "error" in datasets
    assert datasets["cache_miss"] > 100
