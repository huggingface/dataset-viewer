# Datasets server databases migrations

> Scripts to migrate the datasets server databases

## Configuration

The script can be configured using environment variables. They are grouped by scope.

### Migration script

Set environment variables to configure the job (`DATABASE_MIGRATIONS_` prefix):

- `DATABASE_MIGRATIONS_MONGO_DATABASE`: the name of the database used for storing the migrations history. Defaults to `"datasets_server_maintenance"`.
- `DATABASE_MIGRATIONS_MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Launch

```shell
make run
```
