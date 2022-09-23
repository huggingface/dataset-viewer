# Datasets server API

> API to get the first rows of 🤗 datasets

## Configuration

Set environment variables to configure the following aspects:

- `API_HOSTNAME`: the hostname used by the API endpoint. Defaults to `"localhost"`.
- `API_NUM_WORKERS`: the number of workers of the API endpoint. Defaults to `2`.
- `API_PORT`: the port used by the API endpoint. Defaults to `8000`.
- `ASSETS_DIRECTORY`: directory where the asset files are stored. Defaults to empty, in which case the assets are located in the `datasets_server_assets` subdirectory inside the OS default cache directory.
- `HF_AUTH_PATH`: the path of the external authentication service, on the hub (see `HF_ENDPOINT`). The string must contain `%s` which will be replaced with the dataset name. The external authentication service must return 200, 401, 403 or 404. If empty, the authentication is disabled. Defaults to "/api/datasets/%s/auth-check".
- `HF_ENDPOINT`: URL of the HuggingFace Hub. Defaults to `https://huggingface.co`.
- `HF_TOKEN`: App Access Token (ask moonlanding administrators to get one, only the `read` role is required), to access the gated datasets. Defaults to empty.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_AGE_LONG_SECONDS`: number of seconds to set in the `max-age` header on data endpoints. Defaults to `120` (2 minutes).
- `MAX_AGE_SHORT_SECONDS`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `10` (10 seconds).
- `MONGO_CACHE_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_server_cache"`.
- `MONGO_QUEUE_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_server_queue"`.
- `MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.
- `PROMETHEUS_MULTIPROC_DIR`: the directory where the uvicorn workers share their prometheus metrics. See https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn. Defaults to empty, in which case every worker manages its own metrics, and the /metrics endpoint returns the metrics of a random worker.

## Endpoints

See https://huggingface.co/docs/datasets-server

- /healthcheck: ensure the app is running
- /valid: give the list of the valid datasets
- /is-valid: tell if a dataset is valid
- /webhook: add, update or remove a dataset
- /splits: list the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset
- /first-rows: extract the first [rows](https://huggingface.co/docs/datasets/splits.html) for a dataset split
- /assets: return a static asset, ej. https://datasets-server.huggingface.co/assets/food101/--/default/train/0/image/2885220.jpg
- /metrics: return a list of metrics in the Prometheus format
