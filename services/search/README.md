# Datasets server API - search service

> /search endpoint
> /filter endpoint

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### Duckdb index full text search
- `DUCKDB_INDEX_CACHE_DIRECTORY`: directory where the temporal duckdb index files are downloaded. Defaults to empty.
- `DUCKDB_INDEX_TARGET_REVISION`: the git revision of the dataset where the index file is stored in the dataset repository.

### API service

See [../../libs/libapi/README.md](../../libs/libapi/README.md) for more information about the API configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

See https://huggingface.co/docs/datasets-server

- /healthcheck: ensure the app is running
- /metrics: return a list of metrics in the Prometheus format
- /search: get a slice of a search result over a dataset split
- /filter: filter rows of a dataset split
