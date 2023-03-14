# Datasets server API

> API to get the first rows of 🤗 datasets

## Configuration

The worker can be configured using environment variables. They are grouped by scope.

### API service

Set environment variables to configure the application (`API_` prefix):

- `API_HF_AUTH_PATH`: the path of the external authentication service, on the hub (see `HF_ENDPOINT`). The string must contain `%s` which will be replaced with the dataset name. The external authentication service must return 200, 401, 403 or 404. Defaults to "/api/datasets/%s/auth-check".
- `API_HF_JWT_PUBLIC_KEY_URL`: the URL where the "Hub JWT public key" is published. The "Hub JWT public key" must be in JWK format. It helps to decode a JWT sent by the Hugging Face Hub, for example, to bypass the external authentication check (JWT in the 'X-Api-Key' header). If not set, the JWT are ignored. Defaults to empty.
- `API_HF_JWT_ALGORITHM`: the algorithm used to encode the JWT. Defaults to `"EdDSA"`.
- `API_HF_TIMEOUT_SECONDS`: the timeout in seconds for the requests to the Hugging Face Hub. Defaults to `0.2` (200 ms).
- `API_MAX_AGE_LONG`: number of seconds to set in the `max-age` header on data endpoints. Defaults to `120` (2 minutes).
- `API_MAX_AGE_SHORT`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `10` (10 seconds).

### Uvicorn

The following environment variables are used to configure the Uvicorn server (`API_UVICORN_` prefix):

- `API_UVICORN_HOSTNAME`: the hostname. Defaults to `"localhost"`.
- `API_UVICORN_NUM_WORKERS`: the number of uvicorn workers. Defaults to `2`.
- `API_UVICORN_PORT`: the port. Defaults to `8000`.

### Prometheus

- `PROMETHEUS_MULTIPROC_DIR`: the directory where the uvicorn workers share their prometheus metrics. See https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn. Defaults to empty, in which case every worker manages its own metrics, and the /metrics endpoint returns the metrics of a random worker.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

See https://huggingface.co/docs/datasets-server

- /healthcheck: ensure the app is running
- /valid: give the list of the valid datasets
- /is-valid: tell if a dataset is valid
- /webhook: add, update or remove a dataset
- /splits: list the [splits](https://huggingface.co/docs/datasets/splits.html) names for a dataset
- /first-rows: extract the first [rows](https://huggingface.co/docs/datasets/splits.html) for a dataset split
- /parquet: list the parquet files auto-converted for a dataset
- /metrics: return a list of metrics in the Prometheus format
