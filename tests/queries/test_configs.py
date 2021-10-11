from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.queries.configs import get_configs


def test_get_configs() -> None:
    dataset = "acronym_identification"
    response = get_configs(dataset)
    assert "configs" in response
    configs = response["configs"]
    assert len(configs) == 1
    config = configs[0]
    assert "dataset" in config
    assert config["dataset"] == dataset
    assert "config" in config
    assert config["config"] == DEFAULT_CONFIG_NAME
