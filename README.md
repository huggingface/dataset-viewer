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
- `DATASETS_REVISION`: git reference for the canonical datasets on https://github.com/huggingface/datasets. Defaults to `master`.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_AGE_LONG_SECONDS`: number of seconds to set in the `max-age` header on data endpoints. Defaults to `120` (2 minutes).
- `MAX_AGE_SHORT_SECONDS`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `10` (10 seconds).
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
- the number of started jobs for the same dataset is under `MAX_JOBS_PER_DATASET`

Also specify `HF_TOKEN` with an App Access Token (ask moonlanding administrators to get one, only the `read` role is required) to allow the worker to download gated models from the hub. Defaults to empty.

Also specify `MAX_SIZE_FALLBACK` with the maximum size in bytes of the dataset to fallback in normal mode if streaming fails. Note that it requires to have the size in the info metadata. Set to `0` to disable the fallback. Defaults to `100_000_000`.

`ROWS_MIN_NUMBER` is the min number (defaults to `10`) and `ROWS_MAX_NUMBER` the max number (defaults to `100`) of rows fetched by the worker for the split, and provided in the /rows endpoint response. `ROWS_MAX_BYTES` is the max size of the /rows endpoint response in bytes. Defaults to `1_000_000` (1 MB).

The `WORKER_QUEUE` variable specifies which jobs queue the worker will pull jobs from. It can be equal to `datasets` (default) or `splits`. The `datasets` jobs should be a lot faster than the `splits` ones, so that we should need a lot more workers for `splits` than for `datasets`.

To warm the cache, ie. add all the missing Hugging Face datasets to the queue:

```bash
make warm
```

To empty the databases:

```bash
make clean
```

or individually:

```bash
make clean-cache
make clean-queues         # delete all the jobs
```

See also:

```bash
make cancel-started-jobs
make cancel-waiting-jobs
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
  "datasets": { "empty": 0, "error": 0, "stalled": 0, "valid": 1 },
  "splits": { "empty": 1, "error": 1, "stalled": 0, "valid": 0 },
  "created_at": "2022-01-20T14:40:50Z"
}
```

Beware: a "dataset" is considered valid if it has fetched correctly the configs and splits. The splits themselves can have errors (ie: the rows or columns might have errors)

### /cache-reports

> Give detailed reports on the content of the cache

Example: https://datasets-preview.huggingface.tech/cache-reports

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which the dataset cache reports, with the following structure:

```json
{
  "datasets": {
    "empty": [],
    "error": [],
    "stalled": [],
    "valid": [{ "dataset": "sent_comp", "status": "VALID", "error": null }]
  },
  "splits": {
    "empty": [
      {
        "dataset": "sent_comp",
        "config": "default",
        "split": "train",
        "status": "EMPTY",
        "error": null
      }
    ],
    "error": [
      {
        "dataset": "sent_comp",
        "config": "default",
        "split": "validation",
        "status": "error",
        "error": {
          "status_code": 400,
          "exception": "Status400Error",
          "message": "Cannot get the first rows for the split.",
          "cause_exception": "FileNotFoundError",
          "cause_message": "[Errno 2] No such file or directory: 'https://github.com/google-research-datasets/sentence-compression/raw/master/data/comp-data.eval.json.gz'",
          "cause_traceback": [
            "Traceback (most recent call last):\n",
            "  File \"/home/slesage/hf/datasets-preview-backend/src/datasets_preview_backend/models/row.py\", line 61, in get_rows\n    rows = extract_rows(dataset_name, config_name, split_name, num_rows, hf_token)\n",
            "  File \"/home/slesage/hf/datasets-preview-backend/src/datasets_preview_backend/models/row.py\", line 32, in decorator\n    return func(*args, **kwargs)\n",
            "  File \"/home/slesage/hf/datasets-preview-backend/src/datasets_preview_backend/models/row.py\", line 55, in extract_rows\n    return list(iterable_dataset.take(num_rows))\n",
            "  File \"/home/slesage/hf/datasets-preview-backend/.venv/lib/python3.9/site-packages/datasets/iterable_dataset.py\", line 341, in __iter__\n    for key, example in self._iter():\n",
            "  File \"/home/slesage/hf/datasets-preview-backend/.venv/lib/python3.9/site-packages/datasets/iterable_dataset.py\", line 338, in _iter\n    yield from ex_iterable\n",
            "  File \"/home/slesage/hf/datasets-preview-backend/.venv/lib/python3.9/site-packages/datasets/iterable_dataset.py\", line 273, in __iter__\n    yield from islice(self.ex_iterable, self.n)\n",
            "  File \"/home/slesage/hf/datasets-preview-backend/.venv/lib/python3.9/site-packages/datasets/iterable_dataset.py\", line 78, in __iter__\n    for key, example in self.generate_examples_fn(**self.kwargs):\n",
            "  File \"/home/slesage/.cache/huggingface/modules/datasets_modules/datasets/sent_comp/512501fef5db888ec620cb9e4943420ea7c7c244c60de9222fb50bca1232f4b5/sent_comp.py\", line 136, in _generate_examples\n    with gzip.open(filepath, mode=\"rt\", encoding=\"utf-8\") as f:\n",
            "  File \"/home/slesage/.pyenv/versions/3.9.6/lib/python3.9/gzip.py\", line 58, in open\n    binary_file = GzipFile(filename, gz_mode, compresslevel)\n",
            "  File \"/home/slesage/.pyenv/versions/3.9.6/lib/python3.9/gzip.py\", line 173, in __init__\n    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')\n",
            "FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/google-research-datasets/sentence-compression/raw/master/data/comp-data.eval.json.gz'\n"
          ]
        }
      }
    ],
    "stalled": [],
    "valid": []
  },
  "created_at": "2022-01-20T14:40:27Z"
}
```

Beware: a "dataset" is considered valid if it has fetched correctly the configs and splits. The splits themselves can have errors (ie: the rows or columns might have errors)

### /valid

> Give the list of the valid datasets. Here, a dataset is considered valid if `/splits` returns a valid response, and if `/rows` returns a valid response for _at least one split_. Note that stalled cache entries are considered valid.

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

### /is-valid

> Tells if a dataset is valid. A dataset is considered valid if `/splits` returns a valid response, and if `/rows` returns a valid response for _at least one split_. Note that stalled cache entries are considered valid.

Example: https://datasets-preview.huggingface.tech/is-valid?dataset=glue

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID

Responses:

- `200`: JSON content which tells if the dataset is valid or not

```json
{
  "valid": true
}
```

### /hf-datasets-count-by-cache-status

> Give statistics about the datasets of the hub

Example: https://datasets-preview.huggingface.tech/hf-datasets-count-by-cache-status

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which gives statistics about the status of the public datasets of the Hub, split by canonical or community, with the following structure:

```json
{
  "canonical": {
    "valid": 0,
    "error": 0,
    "missing": 1
  },
  "community": {
    "valid": 0,
    "error": 0,
    "missing": 1
  },
  "created_at": "2022-01-20T13:52:05Z"
}
```

The meaning is the following:

- "valid": the list of splits and the 100 first rows of every split are available (maybe stalled)
- "error": the list of splits could not be fetched, or the rows could not be fetched for some splits
- "missing": the list of splits is missing, or the rows are missing for some splits

### /queue

> Give statistics about the content of the queue

Example: https://datasets-preview.huggingface.tech/queue

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which gives statistics about the queue, with the following structure:

```json
{
  "datasets": {
    "waiting": 0,
    "started": 0,
    "success": 1,
    "error": 0,
    "cancelled": 0
  },
  "splits": {
    "waiting": 0,
    "started": 0,
    "success": 0,
    "error": 0,
    "cancelled": 34
  },
  "created_at": "2022-01-20T13:52:05Z"
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
  "datasets": {
    "waiting": [],
    "started": [],
    "success": [
      {
        "dataset_name": "glue",
        "status": "SUCCESS",
        "created_at": "2022-01-20T13:48:06.705000",
        "started_at": "2022-01-20T13:48:21.615000",
        "finished_at": "2022-01-20T13:48:27.898000"
      }
    ],
    "error": [],
    "cancelled": []
  },
  "splits": {
    "waiting": [],
    "started": [],
    "success": [],
    "error": [],
    "cancelled": [
      {
        "dataset_name": "glue",
        "config_name": "cola",
        "split_name": "test",
        "status": "CANCELLED",
        "created_at": "2022-01-20T13:48:27.846000",
        "started_at": null,
        "finished_at": "2022-01-20T13:51:51.411000"
      }
    ]
  },
  "created_at": "2022-01-20T13:59:03Z"
}
```

### /queue-dump-waiting-started

> Give the queue entries, classed by status, only for "waiting" and "started" statuses

Example: https://datasets-preview.huggingface.tech/queue-dump-waiting-started

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which the queue content, by status, with the following structure:

```json
{
  "datasets": {
    "waiting": [],
    "started": []
  },
  "splits": {
    "waiting": [],
    "started": []
  },
  "created_at": "2022-01-20T13:59:03Z"
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

The three keys are optional, and moonlanding should send only one of them. The dataset identifiers are full names, ie. they must include the `datasets/` prefix, which means that a community dataset will have two slashes: `datasets/allenai/c4` for example.

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

### /refresh-split

> Refresh the cache of rows and columns of a split

Example: https://datasets-preview.huggingface.tech/refresh-split

Method: `POST`

Body:

```json
{
  "dataset": "glue",
  "config": "ax",
  "split": "test"
}
```

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "status": "ok"
  }
  ```

- `400`: the payload is erroneous, or a 400 error raised during the cache operation
- `500`: application error

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

> Lists the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset

Example: https://datasets-preview.huggingface.tech/splits?dataset=glue

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID

Responses:

- `200`: JSON content with the following structure:

  ```json
  {
    "splits": [
      {
        "dataset": "glue",
        "config": "cola",
        "split": "test",
        "num_bytes": 217556,
        "num_examples": 1821
      },
      {
        "dataset": "glue",
        "config": "cola",
        "split": "train",
        "num_bytes": 4715283,
        "num_examples": 67349
      },
      {
        "dataset": "glue",
        "config": "cola",
        "split": "validation",
        "num_bytes": 106692,
        "num_examples": 872
      }
    ]
  }
  ```

- `400`: the dataset script is erroneous
- `404`: the dataset or config cannot be found, or it's not in the cache
- `500`: application error

Note that the value of `"num_bytes"` and `"num_examples"` is set to `null` if the data is not available.

### /rows

> Extract the first [rows](https://huggingface.co/docs/datasets/splits.html) for a split of a dataset config

Example: https://datasets-preview.huggingface.tech/rows?dataset=glue&config=ax&split=test

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID
- `config` (required): the configuration name
- `split` (required): the split name

Responses:

- `200`: JSON content that provides the types of the columns (see features at https://huggingface.co/docs/datasets/about_dataset_features.html) and the data rows, with the following structure. Note that the features are ordered and this order can be used to display the columns in a table for example. Binary values are transmitted in UTF-8 encoded base64 strings. The number of rows depends on `ROWS_MAX_BYTES`, `ROWS_MIN_NUMBER` and `ROWS_MAX_NUMBER`. Note that the content of a cell might be truncated to fit within the limits, in which case the `truncated_cells` array will contain the name of the cell (see the last element in the example), and the cell content will always be a string.

  ```json
  {
    "columns": [
      {
        "dataset": "glue",
        "config": "ax",
        "split": "test",
        "column_idx": 0,
        "column": { "name": "premise", "type": "STRING" }
      },
      {
        "dataset": "glue",
        "config": "ax",
        "split": "test",
        "column_idx": 1,
        "column": { "name": "hypothesis", "type": "STRING" }
      },
      {
        "dataset": "glue",
        "config": "ax",
        "split": "test",
        "column_idx": 2,
        "column": {
          "name": "label",
          "type": "CLASS_LABEL",
          "labels": ["entailment", "neutral", "contradiction"]
        }
      },
      {
        "dataset": "glue",
        "config": "ax",
        "split": "test",
        "column_idx": 3,
        "column": { "name": "idx", "type": "INT" }
      }
    ],
    "rows": [
      {
        "dataset": "glue",
        "config": "ax",
        "split": "test",
        "row_idx": 0,
        "row": {
          "premise": "The cat sat on the mat.",
          "hypothesis": "The cat did not sit on the mat.",
          "label": -1,
          "idx": 0
        },
        "truncated_cells": []
      },
      {
        "dataset": "glue",
        "config": "ax",
        "split": "test",
        "row_idx": 1,
        "row": {
          "premise": "The cat did not sit on the mat.",
          "hypothesis": "The cat sat on the mat.",
          "label": -1,
          "idx": 1
        },
        "truncated_cells": []
      },
      {
        "dataset": "glue",
        "config": "ax",
        "split": "test",
        "row_idx": 2,
        "row": {
          "premise": "When you've got no snow, it's really hard to learn a snow sport so we lo",
          "hypothesis": "When you've got snow, it's really hard to learn a snow sport so we looke",
          "label": -1,
          "idx": 2
        },
        "truncated_cells": ["premise", "hypothesis"]
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
