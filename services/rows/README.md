# Dataset viewer API - rows endpoint

> **GET** /rows

See [usage](https://huggingface.co/docs/dataset-viewer/rows) for more details.

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### API service

See [../../libs/libapi/README.md](../../libs/libapi/README.md) for more information about the API configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

See https://huggingface.co/docs/dataset-viewer

- /healthcheck: ensure the app is running
- /metrics: return a list of metrics in the Prometheus format
- /rows: get a slice of rows of a dataset split
