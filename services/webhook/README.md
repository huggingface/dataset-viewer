# Dataset viewer API - webhook endpoint

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### API service

See [../../libs/libapi/README.md](../../libs/libapi/README.md) for more information about the API configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

See https://huggingface.co/docs/dataset-viewer

- /healthcheck: Ensure the app is running
- /metrics: Return a list of metrics in the Prometheus format
- /webhook: Add, update or remove a dataset
