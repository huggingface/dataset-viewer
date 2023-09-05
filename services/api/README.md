# Datasets server API

> API on 🤗 datasets

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### API service

See [../../libs/libapi/README.md](../../libs/libapi/README.md) for more information about the API configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## /hub-cache endpoint

The `/hub-cache` endpoint is used to cache the datasets' metadata from the Hugging Face Hub.

Set environment variables to configure the application (`HUB_CACHE_` prefix):

- `HUB_CACHE_BASE_URL`: the base URL of the Datasets Server. Defaults to `https://datasets-server.huggingface.co`.
- `HUB_CACHE_NUM_RESULTS_PER_PAGE`: the number of results per page. Defaults to `1_000`.

## Endpoints

See https://huggingface.co/docs/datasets-server

- /healthcheck: Ensure the app is running
- /metrics: Return a list of metrics in the Prometheus format
- /webhook: Add, update or remove a dataset
- /valid: Give the list of the [valid](https://huggingface.co/docs/datasets-server/valid) datasets
- /is-valid: Tell if a dataset is [valid](https://huggingface.co/docs/datasets-server/valid)
- /splits: List the [splits](https://huggingface.co/docs/datasets-server/splits) names for a dataset
- /first-rows: Extract the [first rows](https://huggingface.co/docs/datasets-server/first_rows) for a dataset split
- /parquet: List the [parquet files](https://huggingface.co/docs/datasets-server/parquet) auto-converted for a dataset
