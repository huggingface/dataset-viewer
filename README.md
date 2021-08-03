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
