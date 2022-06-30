# Datasets server admin machine

> Admin scripts

## Install

See [INSTALL](./INSTALL.md#Install)

## Run the scripts

Launch the scripts with:

```shell
make <SCRIPT>
```

Set environment variables to configure the following aspects:

- `ASSETS_DIRECTORY`: directory where the asset files are stored. Defaults to empty, in which case the assets are located in the `datasets_server_assets` subdirectory inside the OS default cache directory.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MONGO_CACHE_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_server_cache"`.
- `MONGO_QUEUE_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_server_queue"`.
- `MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.

To launch the scripts:

- if the image runs in a docker container:

  ```shell
  docker exec -it datasets-server_admin_1 make <SCRIPT>
  ```

- if the image runs in a kube pod:

  ```shell
  kubectl exec datasets-server-prod-admin-5cc8f8fcd7-k7jfc -- make <SCRIPT>
  ```

The scripts:

- `cancel-started-split-jobs`: cancel all the started split jobs (stop the workers before!)
- `cancel-started-dataset-jobs`: cancel all the started dataset jobs (stop the workers before!)
- `cancel-started-splits-jobs`: cancel all the started splits/ jobs (stop the workers before!)
- `cancel-started-first-rows-jobs`: cancel all the started first-rows/ jobs (stop the workers before!)
- `warm-cache`: create jobs for all the missing datasets and/or splits

## Run the API

The admin service provides technical endpoints:

- `/healthcheck`
- `/metrics`: gives info about the cache and the queue
