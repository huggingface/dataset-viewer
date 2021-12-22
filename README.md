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
- `DATASETS_ENABLE_PRIVATE`: enable private datasets. Defaults to `False`.
- `DATASETS_REVISION`: git reference for the canonical datasets on https://github.com/huggingface/datasets. Defaults to `1.17.0`.
- `EXTRACT_ROWS_LIMIT`: number of rows in the extract, if not specified in the API request. Defaults to `100`.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_AGE_LONG_SECONDS`: number of seconds to set in the `max-age` header on data endpoints. Defaults to `21600` (6 hours).
- `MAX_AGE_SHORT_SECONDS`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `120` (2 minutes).
- `MONGO_CACHE_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_preview_cache"`.
- `MONGO_QUEUE_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_preview_queue"`.
- `MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27018"`.
- `WEB_CONCURRENCY`: the number of workers. For now, it's ignored and hardcoded to 1 because the cache is not shared yet. Defaults to `2`.

For example:

```bash
APP_PORT=80 WEB_CONCURRENCY=4 make run
```

To reload the application on file changes while developing, run:

```bash
make watch
```

To launch a worker, which will take jobs from the queue:

```bash
MAX_LOAD_PCT=50 MAX_MEMORY_PCT=60 WORKER_SLEEP_SECONDS=5 make worker
```

Every `WORKER_SLEEP_SECONDS` (defaults to 5 seconds) when idle, the worker will check if resources are available, and update the cache entry for a dataset, if it could get a job from the queue. Then loop to start again. The resources are considered available if all the conditions are met:

- the load percentage (the max of the 1m/5m/15m load divided by the number of cpus \*100) is below `MAX_LOAD_PCT` (defaults to 50%)
- the memory (RAM + SWAP) on the machine is below `MAX_MEMORY_PCT` (defaults to 60%)

To warm the cache, ie. add all the missing Hugging Face datasets to the queue:

```bash
make warm
```

To refresh random 3% of the Hugging Face datasets:

```bash
REFRESH_PCT=3 make refresh
```

The number of randomly chosen datasets to refresh is set by `REFRESH_PCT` (defaults to 1% - set to `100` to refresh all the datasets).

To empty the databases:

```bash
make clean
```

or individually:

```bash
make clean-cache
make clean-queue
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

> Give statistics about the content of the cache

Example: https://datasets-preview.huggingface.tech/cache

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which gives statistics about the datasets in the cache, with the following structure:

```json
{
  "valid": 5,
  "error": 8,
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
    { "dataset": "food101", "status": "valid", "error": null },
    {
      "dataset": "allenai/c4",
      "status": "error",
      "error": {
        "status_code": 404,
        "exception": "Status404Error",
        "message": "The split for the dataset config could not be found.",
        "cause": "FileNotFoundError",
        "cause_message": "https://huggingface.co/datasets/allenai/c4/resolve/f3b95a11ff318ce8b651afc7eb8e7bd2af469c10/en.noblocklist/c4-train.00000-of-01024.json.gz"
      }
    }
  ],
  "created_at": "2021-10-26T14:13:45Z"
}
```

### /valid

> Give the list of the valid datasets

Example: https://datasets-preview.huggingface.tech/valid

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which gives the list of the datasets per status, with the following structure.

```json
{
  "valid": ["discovery"],
  "created_at": "2021-10-07T13:33:46Z"
}
```

### /queue

> Give statistics about the content of the queue

Example: https://datasets-preview.huggingface.tech/queue

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which gives statistics about the queue, with the following structure:

```json
{
  "waiting": 1,
  "started": 0,
  "done": 0,
  "created_at": "2021-10-26T21:17:31Z"
}
```

### /queue-dump

> Give the queue entries, classed by status

Example: https://datasets-preview.huggingface.tech/queue-dump

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which the queue content, by status, with the following structure:

```json
{
  "waiting": [
    {
      "dataset_name": "acronym_identification",
      "created_at": "2021-10-26T20:55:06.850000",
      "started_at": null,
      "finished_at": null
    }
  ],
  "started": [],
  "finished": [],
  "created_at": "2021-10-29T08:19:20Z"
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
- `500`: application error

Note: if you want to refresh multiple datasets at a time, you have to call the endpoint again and again. You can use bash for example:

```bash
MODELS=(amazon_polarity ami arabic_billion_words)
for model in ${MODELS[@]}; do curl -X POST https://datasets-preview.huggingface.tech/webhook -H 'Content-Type: application/json' -d '{"update": "datasets/'$model'"}'; done;
```

### /hf_datasets

> Lists the HuggingFace [datasets](https://huggingface.co/docs/datasets/loading_datasets.html#selecting-a-configuration): canonical and community

Example: https://datasets-preview.huggingface.tech/hf_datasets

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "datasets": [
      {
        "id": "acronym_identification",
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
        "citation": "@inproceedings{veyseh-et-al-2020-what,\n   title={{What Does This Acronym Mean? Introducing a New Dataset for Acronym Identification and Disambiguation}},\n   author={Amir Pouran Ben Veyseh and Franck Dernoncourt and Quan Hung Tran and Thien Huu Nguyen},\n   year={2020},\n   booktitle={Proceedings of COLING},\n   link={https://arxiv.org/pdf/2010.14678v1.pdf}\n}",
        "description": "Acronym identification training and development sets for the acronym identification task at SDU@AAAI-21.",
        "paperswithcode_id": "acronym-identification",
        "downloads": 5174
      },
      {
        "id": "aeslc",
        "tags": ["languages:en"],
        "citation": "@misc{zhang2019email,\n    title={This Email Could Save Your Life: Introducing the Task of Email Subject Line Generation},\n    author={Rui Zhang and Joel Tetreault},\n    year={2019},\n    eprint={1906.03497},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}\n}",
        "description": "A collection of email messages of employees in the Enron Corporation.\n\nThere are two features:\n  - email_body: email body text.\n  - subject_line: email subject text.",
        "paperswithcode_id": "aeslc",
        "downloads": 3053
      },
      {
        "id": "afrikaans_ner_corpus",
        "tags": [
          "annotations_creators:expert-generated",
          "language_creators:expert-generated",
          "languages:af",
          "licenses:other-Creative Commons Attribution 2.5 South Africa License",
          "multilinguality:monolingual",
          "size_categories:1K<n<10K",
          "source_datasets:original",
          "task_categories:structure-prediction",
          "task_ids:named-entity-recognition"
        ],
        "citation": "@inproceedings{afrikaans_ner_corpus,\n  author    = {\tGerhard van Huyssteen and\n                Martin Puttkammer and\n                E.B. Trollip and\n                J.C. Liversage and\n              Roald Eiselen},\n  title     = {NCHLT Afrikaans Named Entity Annotated Corpus},\n  booktitle = {Eiselen, R. 2016. Government domain named entity recognition for South African languages. Proceedings of the 10th      Language Resource and Evaluation Conference, Portorož, Slovenia.},\n  year      = {2016},\n  url       = {https://repo.sadilar.org/handle/20.500.12185/299},\n}",
        "description": "Named entity annotated data from the NCHLT Text Resource Development: Phase II Project, annotated with PERSON, LOCATION, ORGANISATION and MISCELLANEOUS tags.",
        "paperswithcode_id": null,
        "downloads": 229
      }
    ]
  }
  ```

- `500`: application error

### /splits

> Lists the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset config

Example: https://datasets-preview.huggingface.tech/splits?dataset=glue&config=cola

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
- `404`: the dataset or config cannot be found, or it's not in the cache
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
    "columns": [
      {
        "dataset": "glue",
        "config": "cola",
        "split": "train",
        "column_idx": 0,
        "column": {
          "name": "sentence",
          "type": "STRING"
        }
      },
      {
        "dataset": "glue",
        "config": "cola",
        "split": "train",
        "column_idx": 1,
        "column": {
          "name": "label",
          "type": "CLASS_LABEL",
          "labels": ["unacceptable", "acceptable"]
        }
      },
      {
        "dataset": "glue",
        "config": "cola",
        "split": "train",
        "column_idx": 2,
        "column": {
          "name": "idx",
          "type": "INT"
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
- `404`: the dataset, config or script cannot be found, or it's not in the cache
- `500`: application error

### /assets

> Return an asset

Example: https://datasets-preview.huggingface.tech/assets/food101/--/default/train/0/image/2885220.jpg

Method: `GET`

Path parameters:

`/assets/:dataset/--/:config/:split/:row_idx/:column/:filename`

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
