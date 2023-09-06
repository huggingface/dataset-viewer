# Datasets server SSE API

> Server-sent events API for the Datasets server. It's used to update the Hub's backend cache.

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

See https://huggingface.co/docs/datasets-server

- /healthcheck: Ensure the app is running
- /metrics: Return a list of metrics in the Prometheus format
- /hub-cache: Return a dataset information for Hub's backend cache when a dataset is updated
