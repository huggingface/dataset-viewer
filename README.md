# Datasets preview backend

> API to extract rows of 🤗 datasets

The URL schema is `https://huggingface.co/datasets-preview/:datasetId/extract?rows=100`. For example https://huggingface.co/datasets-preview/acronym_identification/extract?rows=10 will return a JSON file with the list of the first 10 rows of the first available dataset split in https://huggingface.co/datasets/acronym_identification.

## Requirements

- Python 3.8+

## Install

```bash
git clone git@github.com:huggingface/datasets-preview-backend.git
cd datasets-preview-backend
python -m venv .venv
source .venv/bin/activate
pip install .
deactivate
```

See [INSTALL.md](./INSTALL.md) for details on how it has been deployed.

## Run

```bash
cd datasets-preview-backend
source .venv/bin/activate
python datasets-preview-backend/main.py
```

Set environment variables to configure the following aspects:

- `DPB_EXTRACT_ROWS_LIMIT`: maximum number of rows in the extract. Defaults to `100`.
- `DPB_PORT`: the port used by the app
