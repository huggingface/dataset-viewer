# Datasets server admin machine

> Admin scripts and endpoints

## Configuration

Set environment variables to configure the following aspects:

- `ASSETS_DIRECTORY`: directory where the asset files are stored. Defaults to empty, in which case the assets are located in the `datasets_server_assets` subdirectory inside the OS default cache directory.
- `CACHE_REPORTS_NUM_RESULTS`: the number of results in /cache-reports/... endpoints. Defaults to `100`.
- `HF_ENDPOINT`: URL of the HuggingFace Hub. Defaults to `https://huggingface.co`.
- `HF_ORGANIZATION`: the huggingface organization from which the authenticated user must be part of in order to access the protected routes, eg. "huggingface". If empty, the authentication is disabled. Defaults to None.
- `HF_WHOAMI_PATH`: the path of the external whoami service, on the hub (see `HF_ENDPOINT`), eg. "/api/whoami-v2". If empty, the authentication is disabled. Defaults to None.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_AGE_SHORT_SECONDS`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `10` (10 seconds).
- `MONGO_CACHE_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_server_cache"`.
- `MONGO_QUEUE_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_server_queue"`.
- `MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.
- `PROMETHEUS_MULTIPROC_DIR`: the directory where the uvicorn workers share their prometheus metrics. See https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn. Defaults to empty, in which case every worker manages its own metrics, and the /metrics endpoint returns the metrics of a random worker.

## Endpoints

The admin service provides endpoints:

- `/healthcheck`
- `/metrics`: gives info about the cache and the queue
- `/cache-reports`: give detailed reports on the content of the cache
- `/pending-jobs`: give the pending jobs, classed by queue and status (waiting or started)

## Scripts

The scripts:

- `cancel-jobs-splits`: cancel all the started jobs for /splits (stop the workers before!)
- `cancel-jobs-first-rows`: cancel all the started jobs for /first-rows (stop the workers before!)
- `refresh-cache`: add a /splits job for every HF dataset
- `refresh-cache-canonical`: add a /splits job for every HF canonical dataset
- `refresh-cache-errors`: add a /splits job for every erroneous HF dataset

To launch the scripts:

- if the image runs in a docker container:

  ```shell
  docker exec -it datasets-server_admin_1 make <SCRIPT>
  ```

- if the image runs in a kube pod:

  ```shell
  kubectl exec datasets-server-prod-admin-5cc8f8fcd7-k7jfc -- make <SCRIPT>
  ```
