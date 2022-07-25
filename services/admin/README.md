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
- `MAX_AGE_SHORT_SECONDS`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `10` (10 seconds).
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
- `refresh-cache`: add a job for every HF dataset
- `refresh-cache-canonical`: add a job for every HF canonical dataset
- `warm-cache`: create jobs for all the missing datasets and/or splits

## Run the API

The admin service provides technical endpoints, all under the `/admin/` path:

- `/admin/healthcheck`
- `/admin/metrics`: gives info about the cache and the queue
- `/admin/cache-reports`: give detailed reports on the content of the cache
- `/admin/pending-jobs`: give the pending jobs, classed by queue and status (waiting or started)

### /cache-reports

> Give detailed reports on the content of the cache

Example: https://datasets-server.huggingface.co/cache-reports

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content which the dataset cache reports, with the following structure:

```json
{
  "/splits-next": [{ "dataset": "sent_comp", "status": "200", "error": null }],
  "/first-rows": [
      {
        "dataset": "sent_comp",
        "config": "default",
        "split": "validation",
        "status": "400",
        "error": {
          "message": "Cannot get the first rows for the split.",
          "cause_exception": "FileNotFoundError",
        }
      },
      {
        "dataset": "sent_comp",
        "config": "default",
        "split": "test",
        "status": "500",
        "error": {
          "message": "Internal error.",
        }
      }
    ]
  },
  "created_at": "2022-01-20T14:40:27Z"
}
```

### /pending-jobs

> Give the pending jobs, classed by queue and status (waiting or started)

Example: https://datasets-server.huggingface.co/pending-jobs

Method: `GET`

Parameters: none

Responses:

- `200`: JSON content with the jobs by queue and status, with the following structure:

```json
{
  "/splits": {
    "waiting": [],
    "started": []
  },
  "/rows": {
    "waiting": [],
    "started": []
  },
  "/splits-next": {
    "waiting": [],
    "started": []
  },
  "/first-rows": {
    "waiting": [],
    "started": []
  },
  "created_at": "2022-01-20T13:59:03Z"
}
```
