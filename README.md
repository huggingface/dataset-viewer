# Datasets preview backend

> API to extract rows of 🤗 datasets

## Requirements

- Python 3.8+
- Poetry 1.1.7+
- make
- libicu-dev

## Install

Install with:

```bash
make install
```

## Run

Launch with:

```bash
make run
```

Set environment variables to configure the following aspects:

- `APP_HOSTNAME`: the hostname used by the app. Defaults to `"localhost"`.
- `APP_PORT`: the port used by the app. Defaults to `8000`.
- `ASSETS_DIRECTORY`: directory where the asset files are stored. Defaults to empty, in which case the assets are located in the `datasets_preview_backend_assets` subdirectory inside the OS default cache directory.
- `CACHE_DIRECTORY`: directory where the cache is stored (see http://www.grantjenks.com/docs/diskcache/tutorial.html). It's only applied if `CACHE_PERSIST` is `True`. Defaults to empty, in which case the cache is located in the `datasets_preview_backend` subdirectory inside the OS default cache directory.
- `CACHE_PERSIST`: persist the cache between runs by using the platform's default caches directory. Defaults to `True`.
- `CACHE_SIZE_LIMIT`: maximum size of the cache in bytes. Defaults to `10000000000` (10 GB).
- `DATASETS_ENABLE_PRIVATE`: enable private datasets. Defaults to `False`.
- `DATASETS_REVISION`: git reference for the canonical datasets on https://github.com/huggingface/datasets. Defaults to `master`.
- `EXTRACT_ROWS_LIMIT`: number of rows in the extract, if not specified in the API request. Defaults to `100`.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_AGE_LONG_SECONDS`: number of seconds to set in the `max-age` header on data endpoints. Defaults to `21600` (6 hours).
- `MAX_AGE_SHORT_SECONDS`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `120` (2 minutes).
- `WEB_CONCURRENCY`: the number of workers. For now, it's ignored and hardcoded to 1 because the cache is not shared yet. Defaults to `2`.

For example:

```bash
APP_PORT=80 WEB_CONCURRENCY=4 make run
```

To reload the application on file changes while developing, run:

```bash
make watch
```

To warm the cache:

```bash
MAX_LOAD_PCT=50 MAX_VIRTUAL_MEMORY_PCT=95 MAX_SWAP_MEMORY_PCT=80 make warm
```

Cache warming uses only one thread, and before warming a new dataset, it waits until the load percentage, ie. the 1m load divided by the number of cpus \*100, is below `MAX_LOAD_PCT` (defaults to 50%). Also, if the virtual memory on the machine reaches `MAX_VIRTUAL_MEMORY_PCT` (defaults to 95%), or if the swap memory reaches `MAX_SWAP_MEMORY_PCT` (defaults to 60%), the process stops.

To refresh random 3% of the datasets:

```bash
REFRESH_PCT=3 make refresh
```

The number of randomly chosen datasets to refresh is set by `REFRESH_PCT` (defaults to 1% - set to `100` to refresh all the datasets). Same limits as warming apply with `MAX_LOAD_PCT`, `MAX_VIRTUAL_MEMORY_PCT` and `MAX_SWAP_MEMORY_PCT`.

## Endpoints

### /healthcheck

> Ensure the app is running

Example: https://datasets-preview.huggingface.tech/healthcheck

Method: `GET`

Parameters: none

Responses:

- `200`: text content `ok`

### /cache

> Give statistics about the content of the cache

Example: https://datasets-preview.huggingface.tech/cache

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which gives statistics about the datasets in the cache, with the following structure:

```json
{
  "expected": 1628,
  "valid": 5,
  "error": 8,
  "cache_miss": 1615,
  "created_at": "2021-10-11T16:33:08Z"
}
```

### /cache-reports

> Give detailed reports on the content of the cache

Example: https://datasets-preview.huggingface.tech/cache-reports

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which the dataset cache reports, with the following structure:

```json
{
  "reports": [
    {
      "dataset": "acronym_identification",
      "tags": [
        "annotations_creators:expert-generated",
        "language_creators:found",
        "languages:en",
        "licenses:mit",
        "multilinguality:monolingual",
        "size_categories:10K<n<100K",
        "source_datasets:original",
        "task_categories:structure-prediction",
        "task_ids:structure-prediction-other-acronym-identification"
      ],
      "downloads": 4634,
      "status": "cache_miss",
      "error": null
    },
    {
      "dataset": "ade_corpus_v2",
      "tags": [
        "annotations_creators:expert-generated",
        "language_creators:found",
        "languages:en",
        "licenses:unknown",
        "multilinguality:monolingual",
        "size_categories:10K<n<100K",
        "size_categories:1K<n<10K",
        "size_categories:n<1K",
        "source_datasets:original",
        "task_categories:text-classification",
        "task_categories:structure-prediction",
        "task_ids:fact-checking",
        "task_ids:coreference-resolution"
      ],
      "downloads": 3292,
      "status": "cache_miss",
      "error": null
    },
    {
      "dataset": "adversarial_qa",
      "tags": [
        "annotations_creators:crowdsourced",
        "language_creators:found",
        "languages:en",
        "licenses:cc-by-sa-4.0",
        "multilinguality:monolingual",
        "size_categories:10K<n<100K",
        "source_datasets:original",
        "task_categories:question-answering",
        "task_ids:extractive-qa",
        "task_ids:open-domain-qa"
      ],
      "downloads": 40637,
      "status": "cache_miss",
      "error": null
    }
  ],
  "created_at": "2021-10-08T08:27:46Z"
}
```

### /valid

> Give the list of the datasets, by validity status

Example: https://datasets-preview.huggingface.tech/valid

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which gives the list of the datasets per status, with the following structure.

```json
{
  "valid": ["discovery"],
  "error": ["TimTreasure4/Test"],
  "cache_miss": [
    "acronym_identification",
    "ade_corpus_v2",
    "adversarial_qa",
    "aeslc"
  ],
  "created_at": "2021-10-07T13:33:46Z"
}
```

### /webhook

> Adds, updates or removes a cache entry

Example: https://datasets-preview.huggingface.tech/webhook

Method: `POST`

Body:

```json
{
  "add": "datasets/dataset1",
  "update": "datasets/dataset1",
  "remove": "datasets/dataset1"
}
```

The three keys are optional, and moonlanding should send only one of them. `add` and `update` take some time to respond, because the dataset is fetched, while `remove` returns immediately. The dataset identifiers are full names, ie. they must include the `datasets/` prefix, which means that a community dataset will have two slashes: `datasets/allenai/c4` for example.

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "status": "ok"
  }
  ```

- `400`: the payload is erroneous, or a 400 error raised during the cache operation
- `404`: a 404 error raised during the cache operation
- `500`: application error

### /datasets

> Lists the [datasets](https://huggingface.co/docs/datasets/loading_datasets.html#selecting-a-configuration) names: canonical and community

Example: https://datasets-preview.huggingface.tech/datasets

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "datasets": [
      {"dataset": "acronym_identification"},
      {"dataset": "ade_corpus_v2"},
      {"dataset": "adversarial_qa"},
      {"dataset": "aeslc"},
      {"dataset": "afrikaans_ner_corpus"},
      {"dataset": "ag_news"},
      ...
    ]
  }
  ```

- `500`: application error

### /configs

> Lists the [configurations](https://huggingface.co/docs/datasets/loading_datasets.html#selecting-a-configuration) names for the dataset

Example: https://datasets-preview.huggingface.tech/configs?dataset=glue

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "configs": [
      {
        "dataset": "glue",
        "config": "cola"
      },
      {
        "dataset": "glue",
        "config": "sst2"
      },
      {
        "dataset": "glue",
        "config": "mrpc"
      },
      {
        "dataset": "glue",
        "config": "qqp"
      },
      {
        "dataset": "glue",
        "config": "stsb"
      },
      {
        "dataset": "glue",
        "config": "mnli"
      },
      {
        "dataset": "glue",
        "config": "mnli_mismatched"
      },
      {
        "dataset": "glue",
        "config": "mnli_matched"
      },
      {
        "dataset": "glue",
        "config": "qnli"
      },
      {
        "dataset": "glue",
        "config": "rte"
      },
      {
        "dataset": "glue",
        "config": "wnli"
      },
      {
        "dataset": "glue",
        "config": "ax"
      }
    ]
  }
  ```

  Note that if there is only one default config, it will be named `"default"`. See https://datasets-preview.huggingface.tech/configs?dataset=sent_comp for example.

- `400`: the dataset script is erroneous
- `404`: the dataset cannot be found
- `500`: application error

### /infos

> Return the dataset_info.json file for the dataset

Example: https://datasets-preview.huggingface.tech/infos?dataset=glue

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "infos": [
      {
        "dataset": "glue",
        "config": "cola",
        "info": {
          "description": "GLUE, the General Language Understanding Evaluation benchmark\n(https://gluebenchmark.com/) is a collection of resources for training,\nevaluating, and analyzing natural language understanding systems.\n\n",
          "citation": "@article{warstadt2018neural,\n  title={Neural Network Acceptability Judgments},\n  author={Warstadt, Alex and Singh, Amanpreet and Bowman, Samuel R},\n  journal={arXiv preprint arXiv:1805.12471},\n  year={2018}\n}\n@inproceedings{wang2019glue,\n  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},\n  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},\n  note={In the Proceedings of ICLR.},\n  year={2019}\n}\n\nNote that each GLUE dataset has its own citation. Please see the source to see\nthe correct citation for each contained dataset.",
          "homepage": "https://nyu-mll.github.io/CoLA/",
          "license": "",
          "features": {
            "sentence": {
              "dtype": "string",
              "id": null,
              "_type": "Value"
            },
            "label": {
              "num_classes": 2,
              "names": ["unacceptable", "acceptable"],
              "names_file": null,
              "id": null,
              "_type": "ClassLabel"
            },
            "idx": {
              "dtype": "int32",
              "id": null,
              "_type": "Value"
            }
          },
          "post_processed": null,
          "supervised_keys": null,
          "task_templates": null,
          "builder_name": "glue",
          "config_name": "cola",
          "version": {
            "version_str": "1.0.0",
            "description": "",
            "major": 1,
            "minor": 0,
            "patch": 0
          },
          "splits": {
            "test": {
              "name": "test",
              "num_bytes": 61049,
              "num_examples": 1063,
              "dataset_name": "glue"
            },
            "train": {
              "name": "train",
              "num_bytes": 489149,
              "num_examples": 8551,
              "dataset_name": "glue"
            },
            "validation": {
              "name": "validation",
              "num_bytes": 60850,
              "num_examples": 1043,
              "dataset_name": "glue"
            }
          },
          "download_checksums": {
            "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip": {
              "num_bytes": 376971,
              "checksum": "f212fcd832b8f7b435fb991f101abf89f96b933ab400603bf198960dfc32cbff"
            }
          },
          "download_size": 376971,
          "post_processing_size": null,
          "dataset_size": 611048,
          "size_in_bytes": 988019
        }
      },
      {
        "dataset": "glue",
        "config": "sst2",
        "info": {
          "description": "GLUE, the General Language Understanding Evaluation benchmark\n(https://gluebenchmark.com/) is a collection of resources for training,\nevaluating, and analyzing natural language understanding systems.\n\n",
          "citation": "@inproceedings{socher2013recursive,\n  title={Recursive deep models for semantic compositionality over a sentiment treebank},\n  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew and Potts, Christopher},\n  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},\n  pages={1631--1642},\n  year={2013}\n}\n@inproceedings{wang2019glue,\n  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},\n  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},\n  note={In the Proceedings of ICLR.},\n  year={2019}\n}\n\nNote that each GLUE dataset has its own citation. Please see the source to see\nthe correct citation for each contained dataset.",
          "homepage": "https://nlp.stanford.edu/sentiment/index.html",
          "license": "",
          "features": {
            "sentence": {
              "dtype": "string",
              "id": null,
              "_type": "Value"
            },
            "label": {
              "num_classes": 2,
              "names": ["negative", "positive"],
              "names_file": null,
              "id": null,
              "_type": "ClassLabel"
            },
            "idx": {
              "dtype": "int32",
              "id": null,
              "_type": "Value"
            }
          },
          "post_processed": null,
          "supervised_keys": null,
          "task_templates": null,
          "builder_name": "glue",
          "config_name": "sst2",
          "version": {
            "version_str": "1.0.0",
            "description": "",
            "major": 1,
            "minor": 0,
            "patch": 0
          },
          "splits": {
            "test": {
              "name": "test",
              "num_bytes": 217556,
              "num_examples": 1821,
              "dataset_name": "glue"
            },
            "train": {
              "name": "train",
              "num_bytes": 4715283,
              "num_examples": 67349,
              "dataset_name": "glue"
            },
            "validation": {
              "name": "validation",
              "num_bytes": 106692,
              "num_examples": 872,
              "dataset_name": "glue"
            }
          },
          "download_checksums": {
            "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip": {
              "num_bytes": 7439277,
              "checksum": "d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa"
            }
          },
          "download_size": 7439277,
          "post_processing_size": null,
          "dataset_size": 5039531,
          "size_in_bytes": 12478808
        }
      },
      ...
    ]
  }
  ```

- `400`: the dataset script is erroneous
- `404`: the dataset cannot be found
- `500`: application error

### /splits

> Lists the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset config

Example: https://datasets-preview.huggingface.tech/splits?dataset=glue&config=ax

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID
- `config`: the configuration name. If the dataset does not contain configs, you may explicitly pass "config=default". If obviated, return the splits for all the configs of the dataset.

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "splits": [
      {
        "dataset": "glue",
        "config": "cola",
        "split": "test"
      },
      {
        "dataset": "glue",
        "config": "cola",
        "split": "train"
      },
      {
        "dataset": "glue",
        "config": "cola",
        "split": "validation"
      }
    ]
  }
  ```

- `400`: the dataset script is erroneous
- `404`: the dataset or config cannot be found
- `500`: application error

### /rows

> Extract the first [rows](https://huggingface.co/docs/datasets/splits.html) for a split of a dataset config

Example: https://datasets-preview.huggingface.tech/rows?dataset=glue&config=ax&split=test

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID
- `config`: the configuration name. If the dataset does not contain configs, you may explicitly pass "config=default". If obviated, return the rows for all the configs of the dataset.
- `split`: the split name. It's ignored if `config` is empty. If obviated, return the rows for all the splits of the config, or of the dataset if `config` is obviated too.

Responses:

- `200`: JSON content that provides the types of the columns (see features at https://huggingface.co/docs/datasets/about_dataset_features.html) and the data rows, with the following structure. Note that the features are ordered and this order can be used to display the columns in a table for example. Binary values are transmitted in UTF-8 encoded base64 strings.

  ```json
  {
    "features": [
      {
        "dataset": "glue",
        "config": "ax",
        "feature": {
          "name": "premise",
          "content": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
          }
        }
      },
      {
        "dataset": "glue",
        "config": "ax",
        "feature": {
          "name": "hypothesis",
          "content": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
          }
        }
      },
      {
        "dataset": "glue",
        "config": "ax",
        "feature": {
          "name": "label",
          "content": {
            "num_classes": 3,
            "names": ["entailment", "neutral", "contradiction"],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
          }
        }
      },
      {
        "dataset": "glue",
        "config": "ax",
        "feature": {
          "name": "idx",
          "content": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
          }
        }
      }
    ],
    "rows": [
      {
        "dataset": "glue",
        "config": "cola",
        "split": "train",
        "row": {
          "sentence": "Our friends won't buy this analysis, let alone the next one we propose.",
          "label": 1,
          "idx": 0
        }
      },
      {
        "dataset": "glue",
        "config": "cola",
        "split": "train",
        "row": {
          "sentence": "One more pseudo generalization and I'm giving up.",
          "label": 1,
          "idx": 1
        }
      },
      {
        "dataset": "glue",
        "config": "cola",
        "split": "train",
        "row": {
          "sentence": "One more pseudo generalization or I'm giving up.",
          "label": 1,
          "idx": 2
        }
      }
    ]
  }
  ```

- `400`: the dataset script is erroneous, or the data cannot be obtained.
- `404`: the dataset, config or script cannot be found
- `500`: application error

### /assets

> Return an asset

Example: https://datasets-preview.huggingface.tech/assets/food101/___/default/train/0/image/2885220.jpg

Method: `GET`

Path parameters:

`/assets/:dataset/___/:config/:split/:row_idx/:column/:filename`

- `dataset` (required): the dataset ID
- `config` (required): the configuration name. If the dataset does not contain configs, you must explicitly pass "config=default"
- `split` (required): the split name
- `row_idx` (required): the 0-based row index
- `column` (required): the column name
- `filename` (required): the asset file name

Responses:

- `200`: the asset file
- `400`: the dataset script is erroneous, or the data cannot be obtained.
- `404`: the dataset, config, script, row, column, filename or data cannot be found
- `500`: application error
