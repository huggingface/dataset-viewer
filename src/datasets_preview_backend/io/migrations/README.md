# MongoDB migrations

The cache and the queue are stored in two MongoDB databases. They are defined by the env vars: `MONGO_CACHE_DATABASE` and `MONGO_CACHE_DATABASE`, see the [README](../../../../README.md).

When the structure of a database is changed, the data stored in the database must be migrated to the new structure. It's done using the migration scripts in this directory.

## Apply a migration script

The commit, and the release, MUST always give the list of migration scripts that must be applied to migrate.

To run a script, for example [20220406_cache_dbrow_status_and_since.py](./20220406_cache_dbrow_status_and_since.py):

```shell
poetry run python src/datasets_preview_backend/io/migrations/20220406_cache_dbrow_status_and_since.py
```

## Write a migration script

A script filename must contain the date, the database, and a description of the change.

A migration script should apply the changes, then check for the entries to be in a good state. See [20220406_cache_dbrow_status_and_since.py](./20220406_cache_dbrow_status_and_since.py) for example.

See https://docs.mongoengine.org/guide/migration.html for more details on migration scripts with mongoengine.
