from datasets_preview_backend.queries.configs import (
    get_configs,
)


def test_get_configs():
    dataset = "acronym_identification"
    response = get_configs(dataset)
    assert "dataset" in response
    assert response["dataset"] == dataset
    assert "configs" in response
    configs = response["configs"]
    assert len(configs) == 1
    assert configs[0] is None

    configs = get_configs("glue")["configs"]
    assert len(configs) == 12
    assert "cola" in configs


def test_import_nltk():
    # requires the nltk dependency
    configs = get_configs("vershasaxena91/squad_multitask")["configs"]
    assert len(configs) == 3
