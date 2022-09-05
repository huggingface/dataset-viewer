# Datasets server API

> API to get the first rows of 🤗 datasets

## Install

See [INSTALL](./INSTALL.md#Install)

## Run

Launch with:

```bash
make run
```

Set environment variables to configure the following aspects:

- `API_HOSTNAME`: the hostname used by the API endpoint. Defaults to `"localhost"`.
- `API_NUM_WORKERS`: the number of workers of the API endpoint. Defaults to `2`.
- `API_PORT`: the port used by the API endpoint. Defaults to `8000`.
- `ASSETS_DIRECTORY`: directory where the asset files are stored. Defaults to empty, in which case the assets are located in the `datasets_server_assets` subdirectory inside the OS default cache directory.
- `HF_AUTH_PATH`: the path of the external authentication service, on the hub (see `HF_ENDPOINT`). The string must contain `%s` which will be replaced with the dataset name. The external authentication service must return 200, 401, 403 or 404. If empty, the authentication is disabled. Defaults to "/api/datasets/%s/auth-check".
- `HF_ENDPOINT`: URL of the HuggingFace Hub. Defaults to `https://huggingface.co`.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_AGE_LONG_SECONDS`: number of seconds to set in the `max-age` header on data endpoints. Defaults to `120` (2 minutes).
- `MAX_AGE_SHORT_SECONDS`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `10` (10 seconds).
- `MONGO_CACHE_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_server_cache"`.
- `MONGO_QUEUE_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_server_queue"`.
- `MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.
- `PROMETHEUS_MULTIPROC_DIR`: the directory where the uvicorn workers share their prometheus metrics. See https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn. Defaults to empty, in which case every worker manages its own metrics, and the /metrics endpoint returns the metrics of a random worker.

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

Example: https://datasets-server.huggingface.co/healthcheck

Method: `GET`

Parameters: none

Responses:

- `200`: text content `ok`

### /valid

> Give the list of the valid datasets. Here, a dataset is considered valid if `/splits` returns a valid response, and if `/rows` returns a valid response for _at least one split_. Note that stale cache entries are considered valid.

Example: https://datasets-server.huggingface.co/valid

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

> Tells if a dataset is valid. A dataset is considered valid if `/splits` returns a valid response, and if `/rows` returns a valid response for _at least one split_. Note that stale cache entries are considered valid.

Example: https://datasets-server.huggingface.co/is-valid?dataset=glue

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

### /webhook

> Adds, updates or removes a cache entry

Example: https://datasets-server.huggingface.co/webhook

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
for model in ${MODELS[@]}; do curl -X POST https://datasets-server.huggingface.co/webhook -H 'Content-Type: application/json' -d '{"update": "datasets/'$model'"}'; done;
```

### /splits

> Lists the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset

Example: https://datasets-server.huggingface.co/splits?dataset=glue

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

Example: https://datasets-server.huggingface.co/rows?dataset=glue&config=ax&split=test

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

Example: https://datasets-server.huggingface.co/assets/food101/--/default/train/0/image/2885220.jpg

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

### /metrics

> return a list of metrics in the Prometheus format

Example: https://datasets-server.huggingface.co/metrics

Method: `GET`

Parameters: none

Responses:

- `200`: text content in the Prometheus format:

```text
...
# HELP starlette_requests_in_progress Gauge of requests by method and path currently being processed
# TYPE starlette_requests_in_progress gauge
starlette_requests_in_progress{method="GET",path_template="/metrics"} 1.0
# HELP queue_jobs_total Number of jobs in the queue
# TYPE queue_jobs_total gauge
queue_jobs_total{queue="datasets",status="waiting"} 0.0
queue_jobs_total{queue="datasets",status="started"} 0.0
queue_jobs_total{queue="datasets",status="success"} 3.0
queue_jobs_total{queue="datasets",status="error"} 0.0
queue_jobs_total{queue="datasets",status="cancelled"} 0.0
queue_jobs_total{queue="splits",status="waiting"} 0.0
queue_jobs_total{queue="splits",status="started"} 0.0
queue_jobs_total{queue="splits",status="success"} 4.0
queue_jobs_total{queue="splits",status="error"} 0.0
queue_jobs_total{queue="splits",status="cancelled"} 0.0
# HELP cache_entries_total Number of entries in the cache
# TYPE cache_entries_total gauge
cache_entries_total{cache="datasets",status="empty"} 0.0
cache_entries_total{cache="datasets",status="error"} 0.0
cache_entries_total{cache="datasets",status="stale"} 0.0
cache_entries_total{cache="datasets",status="valid"} 1.0
cache_entries_total{cache="splits",status="empty"} 0.0
cache_entries_total{cache="splits",status="error"} 0.0
cache_entries_total{cache="splits",status="stale"} 0.0
cache_entries_total{cache="splits",status="valid"} 2.0
```
