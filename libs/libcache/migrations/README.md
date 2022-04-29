# MongoDB migrations

The cache is stored in a MongoDB database.

When the structure of a database is changed, the data stored in the database must be migrated to the new structure. It's done using the migration scripts in this directory.

## Apply a migration script

The commit, and the release, MUST always give the list of migration scripts that must be applied to migrate.

Before apply the migration script, be sure to **backup** the database, in case of failure.

```shell
mongodump --forceTableScan --uri=mongodb://localhost:27018 --archive=dump.bson
```

To run a script, for example [20220406_cache_dbrow_status_and_since.py](./20220406_cache_dbrow_status_and_since.py):

```shell
export MONGO_CACHE_DATABASE="datasets_preview_queue_test"
export MONGO_URL="mongodb://localhost:27018"
poetry run python libs/libcache/src/libcache/migrations/<YOUR_MIGRATION_FILE>.py
```

Then, validate with

```shell
export MONGO_CACHE_DATABASE="datasets_preview_queue_test"
export MONGO_URL="mongodb://localhost:27018"
poetry run python libs/libcache/src/libcache/migrations/validate.py
```

In case of **error**, restore the database, else remove the dump file

```shell
# only in case of error!
export MONGO_URL="mongodb://localhost:27018"
mongorestore --drop --uri=${MONGO_URL} --archive=dump.bson
```

## Write a migration script

A script filename must contain the date, the database, and a description of the change.

A migration script should apply the changes, then check for the entries to be in a good state. See [20220406_cache_dbrow_status_and_since.py](./20220406_cache_dbrow_status_and_since.py) for example.

See https://docs.mongoengine.org/guide/migration.html for more details on migration scripts with mongoengine.
