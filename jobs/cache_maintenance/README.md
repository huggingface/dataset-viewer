# Datasets server maintenance job

> Job to run maintenance actions on the datasets-server

Available actions:

- `backfill`: backfill the cache (i.e. create jobs to add the missing entries or update the outdated entries)
- `collect-cache-metrics`: compute and store the cache metrics
- `collect-queue-metrics`: compute and store the queue metrics
- `clean-directory`: clean obsolete files/directories for a given path
- `post-messages`: post messages in Hub discussions
- `skip`: do nothing

## Configuration

The script can be configured using environment variables. They are grouped by scope.

- `CACHE_MAINTENANCE_ACTION`: the action to launch, among `backfill`, `collect-cache-metrics`, `collect-queue-metrics`, `clean-directory` and `post-messages`. Defaults to `skip`.

### Backfill job configurations

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for the following configurations:
- Cached Assets
- Assets
- S3
- Cache
- Queue

### Collect Cache job configurations

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for the following configurations:
- Cache

### Collect Queue job configurations

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for the following configurations:
- Queue

### Clean Directory job configurations

- `DIRECTORY_CLEANING_CACHE_DIRECTORY`: directory location to clean.
- `DIRECTORY_CLEANING_SUBFOLDER_PATTERN`: sub folder pattern inside cache directory to delete files.
- `DIRECTORY_CLEANING_EXPIRED_TIME_INTERVAL_SECONDS`: time in seconds after a file is deleted since its last accessed time.

### Post Messages job configurations

Set environment variables to configure the `post-messages` job:

- `DISCUSSIONS_BOT_ASSOCIATED_USER_NAME`: name of the Hub user associated with the Datasets Server bot app.
- `DISCUSSIONS_BOT_TOKEN`: token of the Datasets Server bot used to post messages in Hub discussions.
- `DISCUSSIONS_PARQUET_REVISION`: revision (branch) where the converted Parquet files are stored.


### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Launch

```shell
make run
```
