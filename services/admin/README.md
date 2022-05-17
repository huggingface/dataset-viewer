# Datasets server admin machine

> Admin scripts

## Install

See [INSTALL](./INSTALL.md#Install)

## Run

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

To access the shell:

- if the image runs in a docker container:

  ```shell
  docker exec -it datasets-server_admin_1 sh
  ```

- if the image runs in a kube pod:

  ```shell
  kubectl exec datasets-server-dev-datasets-admin-5cc8f8fcd7-k7jfc -- sh
  ```

Then run one of those:

```shell
make cancel-started-split-jobs   # cancel all the started split jobs (stop the workers before!)
make cancel-started-dataset-jobs # cancel all the started dataset jobs (stop the workers before!)
make warm-cache                  # create jobs for all the missing datasets and/or splits
```
