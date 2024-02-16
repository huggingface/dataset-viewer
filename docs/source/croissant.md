# Get Croissant metadata

Datasets Server automatically generates the metadata in [Croissant](https://github.com/mlcommons/croissant) format (JSON-LD) for every dataset on the Hugging Face Hub. It lists the dataset's name, description, URL, and the distribution of the dataset as Parquet files, including the columns' metadata. The Croissant metadata is available for all the datasets that can be [converted to Parquet format](./parquet#conversion-to-parquet).

## What is Croissant?

Croissant is a metadata format build on top of [schema.org](https://schema.org/) aimed at describing datasets used for machine learning to help indexing, searching and loading them programmatically.

<Tip>

The [specification](https://github.com/mlcommons/croissant/blob/main/docs/croissant-spec.md) is still in early draft status. It may evolve in the future, and backward compatibility is not guaranteed.

</Tip>

## Get the metadata

This guide shows you how to use Datasets Server's `/croissant` endpoint to retrieve the Croissant metadata associated to a dataset.


The `/croissant` endpoint accepts the dataset name as its query parameter:

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/croissant?dataset=ibm/duorc"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
```
</python>
<js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://datasets-server.huggingface.co/croissant?dataset=ibm/duorc",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const result = await response.json();
    return result;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js>
<curl>
```curl
curl https://datasets-server.huggingface.co/croissant?dataset=ibm/duorc \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The endpoint response is a [JSON-LD](https://json-ld.org/) containing the metadata in the Croissant format. For example, the [`ibm/duorc`](https://huggingface.co/datasets/ibm/duorc) dataset has two configurations, `ParaphraseRC` and `SelfRC` (see the [List splits and configurations](./splits) guide for more details about splits and configurations). The metadata links to their Parquet files and describes the type of each of the six columns: `plot_id`, `plot`, `title`, `question_id`, `question`, and `no_answer`:

```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "column": "ml:column",
    "data": { "@id": "ml:data", "@type": "@json" },
    "dataType": { "@id": "ml:dataType", "@type": "@vocab" },
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
    "recordSet": "ml:recordSet",
    "references": "ml:references",
    "regex": "ml:regex",
    "repeated": "ml:repeated",
    "replace": "ml:replace",
    "sc": "https://schema.org/",
    "separator": "ml:separator",
    "source": "ml:source",
    "subField": "ml:subField",
    "transform": "ml:transform"
  },
  "@type": "sc:Dataset",
  "name": "ibm_duorc",
  "description": "ibm/duorc dataset hosted on Hugging Face and contributed by the HF Datasets community",
  "url": "https://huggingface.co/datasets/ibm/duorc",
  "distribution": [
    {
      "@type": "sc:FileObject",
      "name": "repo",
      "description": "The Hugging Face git repository.",
      "contentUrl": "https://huggingface.co/datasets/ibm/duorc/tree/refs%2Fconvert%2Fparquet",
      "encodingFormat": "git+https",
      "sha256": "https://github.com/mlcommons/croissant/issues/80"
    },
    {
      "@type": "sc:FileSet",
      "name": "parquet-files-for-config-ParaphraseRC",
      "containedIn": "repo",
      "encodingFormat": "application/x-parquet",
      "includes": "ParaphraseRC/*/*.parquet"
    },
    {
      "@type": "sc:FileSet",
      "name": "parquet-files-for-config-SelfRC",
      "containedIn": "repo",
      "encodingFormat": "application/x-parquet",
      "includes": "SelfRC/*/*.parquet"
    }
  ],
  "recordSet": [
    {
      "@type": "ml:RecordSet",
      "name": "ParaphraseRC",
      "description": "ibm/duorc - 'ParaphraseRC' subset\n\nAdditional information:\n- 3 splits: train, validation, test\n- 1 skipped column: answers",
      "field": [
        {
          "@type": "ml:Field",
          "name": "plot_id",
          "description": "Column 'plot_id' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-ParaphraseRC",
            "extract": { "column": "plot_id" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "plot",
          "description": "Column 'plot' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-ParaphraseRC",
            "extract": { "column": "plot" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "title",
          "description": "Column 'title' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-ParaphraseRC",
            "extract": { "column": "title" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "question_id",
          "description": "Column 'question_id' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-ParaphraseRC",
            "extract": { "column": "question_id" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "question",
          "description": "Column 'question' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-ParaphraseRC",
            "extract": { "column": "question" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "no_answer",
          "description": "Column 'no_answer' from the Hugging Face parquet file.",
          "dataType": "sc:Boolean",
          "source": {
            "distribution": "parquet-files-for-config-ParaphraseRC",
            "extract": { "column": "no_answer" }
          }
        }
      ]
    },
    {
      "@type": "ml:RecordSet",
      "name": "SelfRC",
      "description": "ibm/duorc - 'SelfRC' subset\n\nAdditional information:\n- 3 splits: train, validation, test\n- 1 skipped column: answers",
      "field": [
        {
          "@type": "ml:Field",
          "name": "plot_id",
          "description": "Column 'plot_id' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-SelfRC",
            "extract": { "column": "plot_id" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "plot",
          "description": "Column 'plot' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-SelfRC",
            "extract": { "column": "plot" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "title",
          "description": "Column 'title' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-SelfRC",
            "extract": { "column": "title" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "question_id",
          "description": "Column 'question_id' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-SelfRC",
            "extract": { "column": "question_id" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "question",
          "description": "Column 'question' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "distribution": "parquet-files-for-config-SelfRC",
            "extract": { "column": "question" }
          }
        },
        {
          "@type": "ml:Field",
          "name": "no_answer",
          "description": "Column 'no_answer' from the Hugging Face parquet file.",
          "dataType": "sc:Boolean",
          "source": {
            "distribution": "parquet-files-for-config-SelfRC",
            "extract": { "column": "no_answer" }
          }
        }
      ]
    }
  ]
}
```

## Load the dataset

To load the dataset, you can use the [mlcroissant](./mlcroissant.md) library. It provides a simple way to load datasets from Croissant metadata.
