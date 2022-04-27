from libmodels.hf_dataset import get_hf_datasets


# get_dataset_names
def test_get_dataset_items() -> None:
    dataset_items = get_hf_datasets()
    assert len(dataset_items) > 1000
    glue_datasets = [dataset for dataset in dataset_items if dataset["id"] == "glue"]
    assert len(glue_datasets) == 1
    glue = glue_datasets[0]
    assert glue["id"] == "glue"
    assert len(glue["tags"]) > 5
    assert "task_categories:text-classification" in glue["tags"]
    assert glue["citation"] is not None
    assert glue["citation"].startswith("@inproceedings")
    assert glue["description"] is not None
    assert glue["description"].startswith("GLUE, the General Language")
    assert glue["paperswithcode_id"] is not None
    assert glue["paperswithcode_id"] == "glue"
    assert glue["downloads"] is not None
    assert glue["downloads"] > 500000
