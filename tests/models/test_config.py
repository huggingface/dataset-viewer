from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.io.cache import cache_directory  # type: ignore
from datasets_preview_backend.models.config import get_config_names


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


# get_config_names
def test_get_config_names() -> None:
    dataset = "acronym_identification"
    configs = get_config_names(dataset)
    assert len(configs) == 1
    assert configs[0] == DEFAULT_CONFIG_NAME

    configs = get_config_names("glue")
    assert len(configs) == 12
    assert "cola" in configs

    # see https://github.com/huggingface/datasets-preview-backend/issues/17
    configs = get_config_names("allenai/c4")
    assert len(configs) == 1
