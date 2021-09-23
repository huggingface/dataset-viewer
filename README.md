# Datasets preview backend

> API to extract rows of 🤗 datasets

## Requirements

- Python 3.8+
- Poetry 1.1.7+
- make

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
- `CACHE_SIZE_LIMIT`: maximum size of the cache in bytes. Defaults to `1073741824` (1 GiB).
- `CACHE_TTL_SECONDS`: number of seconds the entries are kept in cached. Defaults to `604800`.
- `DATASETS_ENABLE_PRIVATE`: enable private datasets. Defaults to `False`.
- `EXTRACT_ROWS_LIMIT`: number of rows in the extract, if not specified in the API request. Defaults to `100`.
- `HF_TOKEN`: the token to be able to download the private datasets from the hub (see https://huggingface.co/settings/token). Defaults to empty.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `WEB_CONCURRENCY`: the number of workers. For now, it's ignored and hardcoded to 1 because the cache is not shared yet. Defaults to `1`.

For example:

```bash
APP_PORT=80 WEB_CONCURRENCY=4 make run
```

To reload the application on file changes while developing, run:

```bash
make watch
```

## Endpoints

### /healthcheck

> Ensure the app is running

Example: https://datasets-preview.huggingface.tech/healthcheck

Method: `GET`

Parameters: none

Responses:

- `200`: text content `ok`

### /cache

> Ensure the app is running

Example: https://datasets-preview.huggingface.tech/cache

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which gives statistics about the cache, with the following structure:

```json
{
  "endpoints": {
    "/datasets": {
      "endpoint": "/datasets",
      "expected": 1,
      "cached": 1,
      "expired": 0,
      "error": 0,
      "valid": 1
    },
    "/info": {
      "endpoint": "/info",
      "expected": 1490,
      "cached": 15,
      "expired": 0,
      "error": 0,
      "valid": 15
    },
    "/configs": {
      "endpoint": "/configs",
      "expected": 1490,
      "cached": 15,
      "expired": 0,
      "error": 0,
      "valid": 15
    },
    "/splits": {
      "endpoint": "/splits",
      "expected": 79,
      "cached": 79,
      "expired": 0,
      "error": 0,
      "valid": 79
    },
    "/rows": {
      "endpoint": "/rows",
      "expected": 127,
      "cached": 127,
      "expired": 0,
      "error": 20,
      "valid": 107
    }
  }
}
```

### /datasets

> Lists the [datasets](https://huggingface.co/docs/datasets/loading_datasets.html#selecting-a-configuration) names: canonical and community

Example: https://datasets-preview.huggingface.tech/datasets

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "type": "all",
    "datasets": [
      "acronym_identification",
      "ade_corpus_v2",
      "adversarial_qa",
      "aeslc",
      "afrikaans_ner_corpus",
      "ag_news",
      ...
    ]
  }
  ```

- `500`: application error

### /info

> Return the dataset_info.json file for the dataset

Example: https://datasets-preview.huggingface.tech/info?dataset=glue

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "dataset": "glue",
    "info": {
      "cola": {
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
            "names": [
              "unacceptable",
              "acceptable"
            ],
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
          "test"  : {
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
      },
      "sst2": { ... },
      ...
    }
  }
  ```

- `400`: the dataset script is erroneous
- `404`: the dataset cannot be found
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
    "dataset": "glue",
    "configs": [
      "cola",
      "sst2",
      "mrpc",
      "qqp",
      "stsb",
      "mnli",
      "mnli_mismatched",
      "mnli_matched",
      "qnli",
      "rte",
      "wnli",
      "ax"
    ]
  }
  ```

  Note that if there is only one default config, it will be named `"default"`. See https://datasets-preview.huggingface.tech/configs?dataset=sent_comp for example.

- `400`: the dataset script is erroneous
- `404`: the dataset cannot be found
- `500`: application error

### /splits

> Lists the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset config

Example: https://datasets-preview.huggingface.tech/splits?dataset=glue&config=ax

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID
- `config`: the configuration name. It might be required, or not, depending on the dataset. If the dataset does not contain configs, the `config` parameter can equivalently be passed with `config=default` or obviated.

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "dataset": "glue",
    "config": "ax",
    "splits": ["test"]
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
- `config`: the configuration name. It might be required, or not, depending on the dataset. If the dataset does not contain configs, the `config` parameter can equivalently be passed with `config=default` or obviated.
- `split` (required): the split name

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "dataset": "glue",
    "config": "ax",
    "split": "test",
    "rows": [
      {
        "idx": 0,
        "hypothesis": "The cat did not sit on the mat.",
        "label": -1,
        "premise": "The cat sat on the mat."
      },
      {
        "idx": 1,
        "hypothesis": "The cat sat on the mat.",
        "label": -1,
        "premise": "The cat did not sit on the mat."
      }
    ]
  }
  ```

- `400`: the dataset script is erroneous, or the data cannot be obtained.
- `404`: the dataset, config or script cannot be found
- `500`: application error
