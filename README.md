# Datasets preview backend

> API to extract rows of 🤗 datasets

The URL schema is `https://huggingface.co/datasets-preview/:datasetId/extract?rows=100`. For example https://huggingface.co/datasets-preview/acronym_identification/extract?rows=10 will return a JSON file with the list of the first 10 rows of the first available dataset split in https://huggingface.co/datasets/acronym_identification.

## Requirements

- Python 3.8+
- Poetry
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

- `EXTRACT_ROWS_LIMIT`: maximum number of rows in the extract. Defaults to `100`.
- `PORT`: the port used by the app. Defaults to `8000`.
- `WEB_CONCURRENCY`: the number of workers. Defaults to `1`.

For example:

```bash
PORT=80 WEB_CONCURRENCY=4 make run
```

To reload the application on file changes while developing, run:

```bash
make watch
```

## Endpoints

### /healthcheck

Endpoint: `/healthcheck`

Example: http://54.158.211.3/healthcheck

> Ensure the app is running

Method: `GET`

Parameters: none

Responses:

- `200`: text content `ok`

### /configs

Endpoint: `/configs`

Example: http://54.158.211.3/configs?dataset=glue

> Lists the [configurations](https://huggingface.co/docs/datasets/loading_datasets.html#selecting-a-configuration) names for the dataset

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

- `400`: the dataset script is erroneous
- `404`: the dataset cannot be found
- `500`: application error

### /splits

Endpoint: `/splits`

Example: http://54.158.211.3/splits?dataset=glue&config=ax

> Lists the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset config

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID
- `config`: the configuration name. It might be required, or not, depending on the dataset

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

Endpoint: `/rows`

Example: http://54.158.211.3/rows?dataset=glue&config=ax&split=test&rows=2

> Extract the first [rows](https://huggingface.co/docs/datasets/splits.html) for a split of a dataset config

Method: `GET`

Parameters:

- `dataset` (required): the dataset ID
- `config`: the configuration name. It might be required, or not, depending on the dataset
- `split` (required): the split name
- `rows`: the number of rows to extract. Defaults to 100.

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
