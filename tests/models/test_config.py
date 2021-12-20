from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.models.config import get_config_names


# get_config_names
def test_get_config_names() -> None:

    dataset = "acronym_identification"
    configs = get_config_names(dataset)
    assert len(configs) == 1
    assert configs[0] == DEFAULT_CONFIG_NAME

    configs = get_config_names("glue")
    assert len(configs) == 12
    assert "cola" in configs

    configs = get_config_names("common_voice")
    assert len(configs) == 60
    assert "tr" in configs
