# Datasets server admin machine

> Admin scripts and endpoints

## Configuration

The worker con be configured using environment variables. They are grouped by scope.

### Admin service

Set environment variables to configure the application (`ADMIN_` prefix):

- `ADMIN_HF_ORGANIZATION`: the huggingface organization from which the authenticated user must be part of in order to access the protected routes, eg. "huggingface". If empty, the authentication is disabled. Defaults to None.
- `ADMIN_CACHE_REPORTS_NUM_RESULTS`: the number of results in /cache-reports/... endpoints. Defaults to `100`.
- `ADMIN_HF_WHOAMI_PATH`: the path of the external whoami service, on the hub (see `HF_ENDPOINT`), eg. "/api/whoami-v2". Defaults to `/api/whoami-v2`.
- `ADMIN_MAX_AGE`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `10` (10 seconds).

### Uvicorn

The following environment variables are used to configure the Uvicorn server (`ADMIN_UVICORN_` prefix):

- `ADMIN_UVICORN_HOSTNAME`: the hostname. Defaults to `"localhost"`.
- `ADMIN_UVICORN_NUM_WORKERS`: the number of uvicorn workers. Defaults to `2`.
- `ADMIN_UVICORN_PORT`: the port. Defaults to `8000`.

### Prometheus

- `PROMETHEUS_MULTIPROC_DIR`: the directory where the uvicorn workers share their prometheus metrics. See https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn. Defaults to empty, in which case every worker manages its own metrics, and the /metrics endpoint returns the metrics of a random worker.

### Cache

See [../../libs/libcache/README.md](../../libs/libcache/README.md) for more information about the cache configuration.

### Queue

See [../../libs/libqueue/README.md](../../libs/libqueue/README.md) for more information about the queue configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

The admin service provides endpoints:

- `/healthcheck`
- `/metrics`: gives info about the cache and the queue
- `/cache-reports`: give detailed reports on the content of the cache:
  - `/cache-reports/splits`
  - `/cache-reports/first-rows`
  - `/cache-reports/parquet`
- `/pending-jobs`: give the pending jobs, classed by queue and status (waiting or started)
- `/force-refresh`: force refresh cache entries. It's a POST endpoint:
  - `/force-refresh/splits?dataset={dataset}`
  - `/force-refresh/first-rows?dataset={dataset}&config={config}&split={split}`
  - `/force-refresh/parquet?dataset={dataset}`

## Scripts

The scripts:

- `cancel-jobs-splits`: cancel all the started jobs for /splits (stop the workers before!)
- `cancel-jobs-first-rows`: cancel all the started jobs for /first-rows (stop the workers before!)
- `cancel-jobs-parquet`: cancel all the started jobs for /parquet (stop the workers before!)

To launch the scripts:

- if the image runs in a docker container:

  ```shell
  docker exec -it datasets-server_admin_1 make <SCRIPT>
  ```

- if the image runs in a kube pod:

  ```shell
  kubectl exec datasets-server-prod-admin-5cc8f8fcd7-k7jfc -- make <SCRIPT>
  ```
