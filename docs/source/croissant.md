# Get Croissant metadata

The dataset viewer automatically generates the metadata in [Croissant](https://github.com/mlcommons/croissant) format (JSON-LD) for every dataset on the Hugging Face Hub. It lists the dataset's name, description, URL, and the distribution of the dataset as Parquet files, including the columns' metadata. The Croissant metadata is available for all the datasets that can be [converted to Parquet format](./parquet#conversion-to-parquet).

## What is Croissant?

Croissant is a metadata format build on top of [schema.org](https://schema.org/) aimed at describing datasets used for machine learning to help indexing, searching and loading them programmatically.

## Get the metadata

This guide shows you how to use [Hugging Face `/croissant` endpoint](https://huggingface.co/docs/hub/api#get-apidatasetsrepoidcroissant) to retrieve the Croissant metadata associated to a dataset.

The `/croissant` endpoint takes the dataset name in the URL, for example for the `ibm/duorc` dataset:

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://huggingface.co/api/datasets/ibm/duorc/croissant"
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
        "https://huggingface.co/api/datasets/ibm/duorc/croissant",
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
curl https://huggingface.co/api/datasets/ibm/duorc/croissant \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

Under the hood it uses the `https://datasets-server.huggingface.co/croissant-crumbs` endpoint and enriches it with the Hub metadata.

The endpoint response is a [JSON-LD](https://json-ld.org/) containing the metadata in the Croissant format. For example, the [`ibm/duorc`](https://huggingface.co/datasets/ibm/duorc) dataset has two subsets, `ParaphraseRC` and `SelfRC` (see the [List splits and subsets](./splits) guide for more details about splits and subsets). The metadata links to their Parquet files and describes the type of each of the six columns: `plot_id`, `plot`, `title`, `question_id`, `question`, and `no_answer`:

```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {
      "@id": "cr:data",
      "@type": "@json"
    },
    "dataBiases": "cr:dataBiases",
    "dataCollection": "cr:dataCollection",
    "dataType": {
      "@id": "cr:dataType",
      "@type": "@vocab"
    },
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
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
    "transform": "cr:transform"
  },
  "@type": "sc:Dataset",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "repo",
      "name": "repo",
      "description": "The Hugging Face git repository.",
      "contentUrl": "https://huggingface.co/datasets/ibm/duorc/tree/refs%2Fconvert%2Fparquet",
      "encodingFormat": "git+https",
      "sha256": "https://github.com/mlcommons/croissant/issues/80"
    },
    {
      "@type": "cr:FileSet",
      "@id": "parquet-files-for-config-ParaphraseRC",
      "name": "parquet-files-for-config-ParaphraseRC",
      "description": "The underlying Parquet files as converted by Hugging Face (see: https://huggingface.co/docs/datasets-server/parquet).",
      "containedIn": {
        "@id": "repo"
      },
      "encodingFormat": "application/x-parquet",
      "includes": "ParaphraseRC/*/*.parquet"
    },
    {
      "@type": "cr:FileSet",
      "@id": "parquet-files-for-config-SelfRC",
      "name": "parquet-files-for-config-SelfRC",
      "description": "The underlying Parquet files as converted by Hugging Face (see: https://huggingface.co/docs/datasets-server/parquet).",
      "containedIn": {
        "@id": "repo"
      },
      "encodingFormat": "application/x-parquet",
      "includes": "SelfRC/*/*.parquet"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "@id": "ParaphraseRC",
      "name": "ParaphraseRC",
      "description": "ibm/duorc - 'ParaphraseRC' subset\n\nAdditional information:\n- 3 splits: train, validation, test\n- 1 skipped column: answers",
      "field": [
        {
          "@type": "cr:Field",
          "@id": "ParaphraseRC/plot_id",
          "name": "ParaphraseRC/plot_id",
          "description": "Column 'plot_id' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-ParaphraseRC"
            },
            "extract": {
              "column": "plot_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "ParaphraseRC/plot",
          "name": "ParaphraseRC/plot",
          "description": "Column 'plot' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-ParaphraseRC"
            },
            "extract": {
              "column": "plot"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "ParaphraseRC/title",
          "name": "ParaphraseRC/title",
          "description": "Column 'title' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-ParaphraseRC"
            },
            "extract": {
              "column": "title"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "ParaphraseRC/question_id",
          "name": "ParaphraseRC/question_id",
          "description": "Column 'question_id' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-ParaphraseRC"
            },
            "extract": {
              "column": "question_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "ParaphraseRC/question",
          "name": "ParaphraseRC/question",
          "description": "Column 'question' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-ParaphraseRC"
            },
            "extract": {
              "column": "question"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "ParaphraseRC/no_answer",
          "name": "ParaphraseRC/no_answer",
          "description": "Column 'no_answer' from the Hugging Face parquet file.",
          "dataType": "sc:Boolean",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-ParaphraseRC"
            },
            "extract": {
              "column": "no_answer"
            }
          }
        }
      ]
    },
    {
      "@type": "cr:RecordSet",
      "@id": "SelfRC",
      "name": "SelfRC",
      "description": "ibm/duorc - 'SelfRC' subset\n\nAdditional information:\n- 3 splits: train, validation, test\n- 1 skipped column: answers",
      "field": [
        {
          "@type": "cr:Field",
          "@id": "SelfRC/plot_id",
          "name": "SelfRC/plot_id",
          "description": "Column 'plot_id' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-SelfRC"
            },
            "extract": {
              "column": "plot_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "SelfRC/plot",
          "name": "SelfRC/plot",
          "description": "Column 'plot' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-SelfRC"
            },
            "extract": {
              "column": "plot"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "SelfRC/title",
          "name": "SelfRC/title",
          "description": "Column 'title' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-SelfRC"
            },
            "extract": {
              "column": "title"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "SelfRC/question_id",
          "name": "SelfRC/question_id",
          "description": "Column 'question_id' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-SelfRC"
            },
            "extract": {
              "column": "question_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "SelfRC/question",
          "name": "SelfRC/question",
          "description": "Column 'question' from the Hugging Face parquet file.",
          "dataType": "sc:Text",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-SelfRC"
            },
            "extract": {
              "column": "question"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "SelfRC/no_answer",
          "name": "SelfRC/no_answer",
          "description": "Column 'no_answer' from the Hugging Face parquet file.",
          "dataType": "sc:Boolean",
          "source": {
            "fileSet": {
              "@id": "parquet-files-for-config-SelfRC"
            },
            "extract": {
              "column": "no_answer"
            }
          }
        }
      ]
    }
  ],
  "name": "duorc",
  "description": "\n\t\n\t\t\n\t\n\t\n\t\tDataset Card for duorc\n\t\n\n\n\t\n\t\t\n\t\n\t\n\t\tDataset Summary\n\t\n\nThe DuoRC dataset is an English language dataset of questions and answers gathered from crowdsourced AMT workers on Wikipedia and IMDb movie plots. The workers were given freedom to pick answer from the plots or synthesize their own answers. It contains two sub-datasets - SelfRC and ParaphraseRC. SelfRC dataset is built on Wikipedia movie plots solely. ParaphraseRC has questions written from Wikipedia movie plots and the… See the full description on the dataset page: https://huggingface.co/datasets/ibm/duorc.",
  "alternateName": [
    "ibm/duorc",
    "DuoRC"
  ],
  "creator": {
    "@type": "Organization",
    "name": "IBM",
    "url": "https://huggingface.co/ibm"
  },
  "keywords": [
    "question-answering",
    "text2text-generation",
    "abstractive-qa",
    "extractive-qa",
    "crowdsourced",
    "crowdsourced",
    "monolingual",
    "100K<n<1M",
    "10K<n<100K",
    "original",
    "English",
    "mit",
    "Croissant",
    "arxiv:1804.07927",
    "🇺🇸 Region: US"
  ],
  "license": "https://choosealicense.com/licenses/mit/",
  "sameAs": "https://duorc.github.io/",
  "url": "https://huggingface.co/datasets/ibm/duorc"
}
```

## Load the dataset

To load the dataset, you can use the [mlcroissant](./mlcroissant) library. It provides a simple way to load datasets from Croissant metadata.
