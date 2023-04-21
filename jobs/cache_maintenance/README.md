# Datasets server maintenance job

> Job to run maintenance actions on the datasets-server

Available actions:

- `backfill`: backfill the cache (i.e. create jobs to add the missing entries or update the outdated entries)
- `metrics`: compute and store the cache and queue metrics
- `skip`: do nothing

## Configuration

The script can be configured using environment variables. They are grouped by scope.

### Actions

Set environment variables to configure the job (`DATABASE_MIGRATIONS_` prefix):

- `CACHE_MAINTENANCE_ACTION`: the action to launch, among `backfill`, `metrics`, `skip`. Defaults to `skip`.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Launch

```shell
make run
```
