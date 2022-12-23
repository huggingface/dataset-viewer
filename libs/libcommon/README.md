# libcommon

A Python library with common code (cache, queue, workers logic, processing steps, configuration, utils, logging, exceptions) used by the services and the workers

## Assets configuration

Set the assets (images and audio files stored locally) environment variables to configure the following aspects:

- `ASSETS_BASE_URL`: base URL for the assets files. Set accordingly to the datasets-server domain, e.g., https://datasets-server.huggingface.co/assets. Defaults to `assets` (TODO: default to an URL).
- `ASSETS_STORAGE_DIRECTORY`: directory where the asset files are stored. Defaults to empty, which means the assets are located in the `datasets_server_assets` subdirectory inside the OS default cache directory.

## Common configuration

Set the common environment variables to configure the following aspects:

- `COMMON_HF_ENDPOINT`: URL of the HuggingFace Hub. Defaults to `https://huggingface.co`.
- `COMMON_HF_TOKEN`: App Access Token (ask moonlanding administrators to get one, only the `read` role is required) to access the gated datasets. Defaults to empty.
- `COMMON_LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. Defaults to `INFO`.

## Cache configuration

Set environment variables to configure the storage of precomputed API responses in a MongoDB database (the "cache"):

- `CACHE_MONGO_DATABASE`: name of the database used for storing the cache. Defaults to `datasets_server_cache`.
- `CACHE_MONGO_URL`: URL used to connect to the MongoDB server. Defaults to `mongodb://localhost:27017`.

## Queue configuration

Set environment variables to configure the job queues to precompute API responses. The job queues are stored in a MongoDB database.

- `QUEUE_MAX_JOBS_PER_NAMESPACE`: maximum number of started jobs for the same namespace (the user or organization, before the `/` separator in the dataset name, or the "canonical" dataset name if not present). Defaults to 1.
- `QUEUE_MONGO_DATABASE`: name of the database used for storing the queue. Defaults to `datasets_server_queue`.
- `QUEUE_MONGO_URL`: URL used to connect to the MongoDB server. Defaults to `mongodb://localhost:27017`.

## Worker loop configuration

Set environment variables to configure the worker loop that processes the queue.

- `WORKER_LOOP_MAX_DISK_USAGE_PCT`: maximum disk usage of every storage disk in the list (in percentage) to allow a job to start. Set to 0 to disable the test. Defaults to 90.
- `WORKER_LOOP_MAX_LOAD_PCT`: maximum load of the machine (in percentage: the max between the 1m load and the 5m load divided by the number of CPUs \*100) allowed to start a job. Set to 0 to disable the test. Defaults to 70.
- `WORKER_LOOP_MAX_MEMORY_PCT`: maximum memory (RAM + SWAP) usage of the machine (in percentage) allowed to start a job. Set to 0 to disable the test. Defaults to 80.
- `WORKER_LOOP_SLEEP_SECONDS`: wait duration in seconds at each loop iteration before checking if resources are available and processing a job if any is available. Note that the loop doesn't wait just after finishing a job: the next job is immediately processed. Defaults to `15`.
- `WORKER_LOOP_STORAGE_PATHS`: comma-separated list of paths to check for disk usage. Defaults to empty.
