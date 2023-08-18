# libapi

A Python library for the API services

## Configuration

The APIs can be configured using environment variables. They are grouped by scope.

### API service

Set environment variables to configure the application (`API_` prefix):

- `API_HF_AUTH_PATH`: the path of the external authentication service, on the hub (see `HF_ENDPOINT`). The string must contain `%s` which will be replaced with the dataset name. The external authentication service must return 200, 401, 403 or 404. Defaults to "/api/datasets/%s/auth-check".
- `API_HF_JWT_PUBLIC_KEY_URL`: the URL where the "Hub JWT public key" is published. The "Hub JWT public key" must be in JWK format. It helps to decode a JWT sent by the Hugging Face Hub, for example, to bypass the external authentication check (JWT in the 'X-Api-Key' header). If not set, the JWT are ignored. Defaults to empty.
- `API_HF_JWT_ALGORITHM`: the algorithm used to encode the JWT. Defaults to `"EdDSA"`.
- `API_HF_TIMEOUT_SECONDS`: the timeout in seconds for the requests to the Hugging Face Hub. Defaults to `0.2` (200 ms).
- `API_HF_WEBHOOK_SECRET`: a shared secret sent by the Hub in the "X-Webhook-Secret" header of POST requests sent to /webhook, to authenticate the originator and bypass some validation of the content (avoiding roundtrip to the Hub). If not set, all the validations are done. Defaults to empty.
- `API_MAX_AGE_LONG`: number of seconds to set in the `max-age` header on data endpoints. Defaults to `120` (2 minutes).
- `API_MAX_AGE_SHORT`: number of seconds to set in the `max-age` header on technical endpoints. Defaults to `10` (10 seconds).

### Uvicorn

The following environment variables are used to configure the Uvicorn server (`API_UVICORN_` prefix):

- `API_UVICORN_HOSTNAME`: the hostname. Defaults to `"localhost"`.
- `API_UVICORN_NUM_WORKERS`: the number of uvicorn workers. Defaults to `2`.
- `API_UVICORN_PORT`: the port. Defaults to `8000`.

### Prometheus

- `PROMETHEUS_MULTIPROC_DIR`: the directory where the uvicorn workers share their prometheus metrics. See https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn. Defaults to empty, in which case every worker manages its own metrics, and the /metrics endpoint returns the metrics of a random worker.
