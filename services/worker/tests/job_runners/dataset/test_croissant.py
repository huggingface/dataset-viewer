# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from unittest.mock import patch

import pytest

from worker.job_runners.dataset.croissant import compute_croissant_response, get_croissant_from_dataset_infos

squad_info = {
    "description": "Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.\n",
    "citation": '@article{2016arXiv160605250R,\n       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},\n                 Konstantin and {Liang}, Percy},\n        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",\n      journal = {arXiv e-prints},\n         year = 2016,\n          eid = {arXiv:1606.05250},\n        pages = {arXiv:1606.05250},\narchivePrefix = {arXiv},\n       eprint = {1606.05250},\n}\n',
    "homepage": "https://rajpurkar.github.io/SQuAD-explorer/",
    "license": ["mit"],
    "tags": ["foo", "doi:hf/123456789", "region:us"],
    "features": {
        "id": {"dtype": "string", "_type": "Value"},
        "title": {"dtype": "string", "_type": "Value"},
        "context": {"dtype": "string", "_type": "Value"},
        "question": {"dtype": "string", "_type": "Value"},
        "answers": {
            "feature": {
                "text": {"dtype": "string", "_type": "Value"},
                "answer_start": {"dtype": "int32", "_type": "Value"},
            },
            "_type": "Sequence",
        },
    },
    "task_templates": [{"task": "question-answering-extractive"}],
    "builder_name": "squad",
    "config_name": "user/squad with space",
    "version": {"version_str": "1.0.0", "description": "", "major": 1, "minor": 0, "patch": 0},
    "splits": {
        "train": {"name": "train", "num_bytes": 79346108, "num_examples": 87599, "dataset_name": "squad"},
        "validation": {
            "name": "validation",
            "num_bytes": 10472984,
            "num_examples": 10570,
            "dataset_name": "squad",
        },
    },
    "download_checksums": {
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json": {
            "num_bytes": 30288272,
            "checksum": None,
        },
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json": {
            "num_bytes": 4854279,
            "checksum": None,
        },
    },
    "download_size": 35142551,
    "dataset_size": 89819092,
    "size_in_bytes": 124961643,
}


def test_get_croissant_from_dataset_infos() -> None:
    croissant = get_croissant_from_dataset_infos("user/squad with space", [squad_info, squad_info], partial=False)
    assert "@context" in croissant
    assert "@type" in croissant
    assert "name" in croissant
    assert croissant["name"] == "user_squad_with_space"
    assert "distribution" in croissant
    assert "recordSet" in croissant
    # column "answers" is not supported (nested)
    assert isinstance(croissant["recordSet"], list)
    assert len(croissant["recordSet"]) == 2
    assert croissant["recordSet"][0]["name"] == "record_set_user_squad_with_space"
    assert croissant["recordSet"][1]["name"] == "record_set_user_squad_with_space_0"
    assert isinstance(croissant["recordSet"][0]["field"], list)
    assert isinstance(squad_info["features"], dict)
    assert len(croissant["recordSet"][0]["field"]) == len(squad_info["features"]) - 1
    assert "1 skipped column: answers" in croissant["recordSet"][0]["description"]
    assert croissant["license"] == ["mit"]
    assert croissant["identifier"] == "hf/123456789"

    # If the parameter doesn't exist, it is not kept:
    del squad_info["license"]
    croissant = get_croissant_from_dataset_infos("user/squad with space", [squad_info, squad_info], partial=False)
    assert "license" not in croissant


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("max_configs", [1, 3])
def test_compute_croissant_response(partial: bool, max_configs: int) -> None:
    with patch(
        "worker.job_runners.dataset.croissant.get_previous_step_or_raise",
        return_value={"content": {"dataset_info": {"a": squad_info, "b": squad_info}, "partial": partial}},
    ), patch("worker.job_runners.dataset.croissant.CROISSANT_MAX_CONFIGS", max_configs):
        croissant_response = compute_croissant_response("user/squad with space")
        assert croissant_response["partial"] == partial
        assert (
            len(croissant_response["croissant"]["recordSet"]) == 1
            if max_configs == 1
            else len(croissant_response["croissant"]["recordSet"]) == 2
        )
        assert croissant_response["truncated_configs"] == (max_configs == 1)
