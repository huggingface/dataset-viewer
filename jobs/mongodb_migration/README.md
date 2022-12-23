# Datasets server databases migrations

> Scripts to migrate the datasets server databases

## Configuration

The script can be configured using environment variables. They are grouped by scope.

### Admin service

Set environment variables to configure the job (`MONGODB_MIGRATION_` prefix):

- `MONGODB_MIGRATION_MONGO_DATABASE`: the name of the database used for storing the migrations history. Defaults to `"datasets_server_maintenance"`.
- `MONGODB_MIGRATION_MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Script

The script:

- `run`: run all the migrations. First look at the previously executed migrations. Run the new ones, and revert them in case of error.

To launch the scripts:

- if the image runs in a docker container:

  ```shell
  docker exec -it datasets-server_mongodb_migration_1 make <SCRIPT>
  ```

- if the image runs in a kube pod:

  ```shell
  kubectl exec datasets-server-prod-mongodb_migration-5cc8f8fcd7-k7jfc -- make <SCRIPT>
  ```
