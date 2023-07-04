# Datasets server API

> API on 🤗 datasets

## Configuration

The worker can be configured using environment variables. They are grouped by scope.

### API service

See [../../libs/libapi/README.md](../../libs/libapi/README.md) for more information about the API configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

See https://huggingface.co/docs/datasets-server

- /healthcheck: ensure the app is running
- /valid: give the list of the valid datasets
- /is-valid: tell if a dataset is valid
- /webhook: add, update or remove a dataset
- /splits: list the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset
- /first-rows: extract the first [rows](https://huggingface.co/docs/datasets/splits.html) for a dataset split
- /parquet: list the parquet files auto-converted for a dataset
- /metrics: return a list of metrics in the Prometheus format
