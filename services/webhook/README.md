# Dataset viewer API - webhook endpoint

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### API service

See [../../libs/libapi/README.md](../../libs/libapi/README.md) for more information about the API configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

### Committer

- `COMMITTER_HF_TOKEN`: the HuggingFace token to commit the parquet/duckdb files to the Hub. The token must be an app token associated with a user that has the right to create/delete ref branches like `refs/converrt/parquet` and `refs/convert/duckdb`.

## Endpoints

See https://huggingface.co/docs/dataset-viewer

- /healthcheck: Ensure the app is running
- /metrics: Return a list of metrics in the Prometheus format
- /webhook: Add, update or remove a dataset
