from unittest.mock import patch

import pytest

from api.routes.croissant import get_croissant_from_dataset_infos

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

v08_context = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "column": "ml:column",
    "data": {"@id": "ml:data", "@type": "@json"},
    "dataBiases": "ml:dataBiases",
    "dataCollection": "ml:dataCollection",
    "dataType": {"@id": "ml:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "extract": "ml:extract",
    "field": "ml:field",
    "fileProperty": "ml:fileProperty",
    "format": "ml:format",
    "includes": "ml:includes",
    "isEnumeration": "ml:isEnumeration",
    "jsonPath": "ml:jsonPath",
    "ml": "http://mlcommons.org/schema/",
    "parentField": "ml:parentField",
    "path": "ml:path",
    "personalSensitiveInformation": "ml:personalSensitiveInformation",
    "recordSet": "ml:recordSet",
    "references": "ml:references",
    "regex": "ml:regex",
    "repeated": "ml:repeated",
    "replace": "ml:replace",
    "sc": "https://schema.org/",
    "separator": "ml:separator",
    "source": "ml:source",
    "subField": "ml:subField",
    "transform": "ml:transform",
}

v1_context = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataBiases": "cr:dataBiases",
    "dataCollection": "cr:dataCollection",
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isEnumeration": "cr:isEnumeration",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "personalSensitiveInformation": "cr:personalSensitiveInformation",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


@pytest.mark.parametrize(["is_v1", "expected_context"], [[True, v1_context], [False, v08_context]])
def test_get_croissant_context_from_dataset_infos(is_v1, expected_context) -> None:
    croissant = get_croissant_from_dataset_infos(
        "user/squad with space",
        [squad_info, squad_info],
        partial=False,
        full_jsonld=False,
        is_v1=is_v1,
    )
    assert croissant["@context"] == expected_context


@pytest.mark.parametrize(["is_v1", "prefix"], [[True, "cr"], [False, "ml"]])
def test_get_croissant_from_dataset_infos(is_v1, prefix):
    croissant = get_croissant_from_dataset_infos(
        "user/squad with space",
        [squad_info, squad_info],
        partial=False,
        full_jsonld=False,
        is_v1=is_v1,
    )
    assert "@context" in croissant
    assert "@type" in croissant
    assert "name" in croissant
    assert croissant["name"] == "user_squad_with_space"

    # Test recordSet.
    assert "recordSet" in croissant
    assert croissant["recordSet"]
    assert isinstance(croissant["recordSet"], list)
    assert len(croissant["recordSet"]) == 2
    assert croissant["recordSet"][0]["@type"] == croissant["recordSet"][1]["@type"] == f"{prefix}:RecordSet"
    assert croissant["recordSet"][0]["name"] == "record_set_user_squad_with_space"
    assert croissant["recordSet"][1]["name"] == "record_set_user_squad_with_space_0"
    assert isinstance(croissant["recordSet"][0]["field"], list)
    assert isinstance(squad_info["features"], dict)
    assert "1 skipped column: answers" in croissant["recordSet"][0]["description"]

    # Test fields.
    assert len(croissant["recordSet"][0]["field"]) == 4
    assert len(croissant["recordSet"][1]["field"]) == 4
    for field in croissant["recordSet"][0]["field"]:
        assert field["@type"] == f"{prefix}:Field"
        assert field["dataType"] == "sc:Text"
    assert len(croissant["recordSet"][0]["field"]) == len(squad_info["features"]) - 1

    # Test distribution.
    assert "distribution" in croissant
    assert croissant["distribution"]
    assert isinstance(croissant["distribution"], list)
    assert croissant["distribution"][0]["@type"] == "sc:FileObject"
    assert croissant["distribution"][1]["@type"] == "sc:FileSet"
    assert croissant["distribution"][2]["@type"] == "sc:FileSet"
    assert croissant["distribution"][0]["name"] == "repo"

    # Test others.
    assert croissant["license"] == ["mit"]
    assert croissant["identifier"] == "hf/123456789"

    # If the parameter doesn't exist, it is not kept:
    squad_licenseless_info = squad_info.copy()
    del squad_licenseless_info["license"]
    croissant = get_croissant_from_dataset_infos(
        "user/squad with space",
        [squad_licenseless_info, squad_licenseless_info],
        partial=False,
        full_jsonld=False,
        is_v1=is_v1,
    )
    assert "license" not in croissant


def test_get_v1_only_croissant_from_dataset_infos():
    croissant = get_croissant_from_dataset_infos(
        "user/squad with space",
        [squad_info, squad_info],
        partial=False,
        full_jsonld=False,
        is_v1=True,
    )
    assert croissant["recordSet"][0]["@id"] == "record_set_user_squad_with_space"
    assert croissant["recordSet"][1]["@id"] == "record_set_user_squad_with_space_0"
    for i, _ in enumerate(croissant["recordSet"]):
        for field in croissant["recordSet"][i]["field"]:
            assert "source" in field
            assert "fileSet" in field["source"]
            assert "@id" in field["source"]["fileSet"]
            assert field["source"]["fileSet"]["@id"]
            assert "extract" in field["source"]
            assert field["source"]["extract"]["column"] == field["@id"]
    for distribution in croissant["distribution"]:
        assert "@id" in distribution
        if "containedIn" in distribution:
            assert "@id" in distribution["containedIn"]


MAX_COLUMNS = 3


@pytest.mark.parametrize(
    ("full_jsonld", "num_columns", "is_v1"),
    [
        (True, 4, True),
        (False, MAX_COLUMNS, False),
    ],
)
def test_get_croissant_from_dataset_infos_max_columns(full_jsonld: bool, num_columns: int, is_v1: bool) -> None:
    with patch("api.routes.croissant.MAX_COLUMNS", MAX_COLUMNS):
        croissant = get_croissant_from_dataset_infos(
            "user/squad with space",
            [squad_info, squad_info],
            partial=False,
            full_jsonld=full_jsonld,
            is_v1=is_v1,
        )
    assert len(croissant["recordSet"][0]["field"]) == num_columns
    assert full_jsonld or "max number of columns reached" in croissant["recordSet"][0]["description"]
