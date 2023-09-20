# Datasets server maintenance job

> Job to run maintenance actions on the datasets-server

Available actions:

- `backfill`: backfill the cache (i.e. create jobs to add the missing entries or update the outdated entries)
- `collect-cache-metrics`: compute and store the cache metrics
- `collect-queue-metrics`: compute and store the queue metrics
- `delete-indexes`: delete temporary DuckDB index files downloaded to handle /search requests
- `post-messages`: post messages in Hub discussions
- `skip`: do nothing

## Configuration

The script can be configured using environment variables. They are grouped by scope.

- `DISCUSSIONS_BOT_ASSOCIATED_USER_NAME`: name of the Hub user associated with the Datasets Server bot app.
- `DISCUSSIONS_BOT_TOKEN`: token of the Datasets Server bot used to post messages in Hub discussions.
- `DISCUSSIONS_PARQUET_REVISION`: revision (branch) where the converted Parquet files are stored.

### Actions

Set environment variables to configure the job (`CACHE_MAINTENANCE_` prefix):

- `CACHE_MAINTENANCE_ACTION`: the action to launch, among `backfill`, `metrics`, `skip`. Defaults to `skip`.

Specific to the backfill action:

- `CACHE_MAINTENANCE_BACKFILL_ERROR_CODES_TO_RETRY`: the list of error codes to retry. Defaults to None.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Launch

```shell
make run
```
