# Datasets server - worker

> Worker to pre-process datasets and splits

## Install

See [INSTALL](./INSTALL.md#Install)

## Run

Launch the worker to preprocess the datasets queue:

```bash
make datasets-worker
```

Launch the worker to preprocess the splits queue:

```bash
make splits-worker
```

Set environment variables to configure the following aspects:

- `ASSETS_DIRECTORY`: directory where the asset files are stored. Defaults to empty, in which case the assets are located in the `datasets_server_assets` subdirectory inside the OS default cache directory.
- `DATASETS_BLOCKLIST`: comma-separated list of datasets that will never be processed. It's used to preventively block the biggest datasets, that we don't know how to manage properly in our infrastructure. An example: `DATASETS_BLOCKLIST="Alvenir/nst-da-16khz,bigscience/P3,clips/mqa"` (use [`\`](https://stackoverflow.com/a/3871336/7351594) to have one dataset per line if it makes the list more readable). Defaults to empty.
- `DATASETS_REVISION`: git reference for the canonical datasets on https://github.com/huggingface/datasets. Defaults to `master`.
- `HF_TOKEN`: App Access Token (ask moonlanding administrators to get one, only the `read` role is required), to access the gated datasets. Defaults to empty.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_JOBS_PER_DATASET`: the maximum number of started jobs for the same dataset. Defaults to 1.
- `MAX_LOAD_PCT`: the maximum load of the machine (in percentage: the max between the 1m load and the 5m load divided by the number of cpus \*100) allowed to start a job. Defaults to 70.
- `MAX_MEMORY_PCT`: the maximum memory (RAM + SWAP) usage of the machine (in percentage) allowed to start a job. Defaults to 80.
- `MAX_SIZE_FALLBACK`: the maximum size in bytes of the dataset to fallback in normal mode if streaming fails. Note that it requires to have the size in the info metadata. Set to `0` to disable the fallback. Defaults to `100_000_000`.
- `MIN_CELL_BYTES`: the minimum size in bytes of a cell when truncating the content of a row (see `ROWS_MAX_BYTES`). Below this limit, the cell content will not be truncated. Defaults to `100`.
- `MONGO_CACHE_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_server_cache"`.
- `MONGO_QUEUE_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_server_queue"`.
- `MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.
- `ROWS_MAX_BYTES`: the max size of the /rows endpoint response in bytes. Defaults to `1_000_000` (1 MB).
- `ROWS_MAX_NUMBER`: the max number of rows fetched by the worker for the split, and provided in the /rows endpoint response. Defaults to `100`.
- `ROWS_MIN_NUMBER`: the min number of rows fetched by the worker for the split, and provided in the /rows endpoint response. Defaults to `10`.
- `WORKER_QUEUE`: name of the queue the worker will pull jobs from. It can be equal to `datasets` or `splits`. The `datasets` jobs should be a lot faster than the `splits` ones, so that we should need a lot more workers for `splits` than for `datasets`. Note that this environment variable is already set to the appropriate value in `make datasets-worker` and `make splits-worker`. Defaults to `datasets`.
- `WORKER_SLEEP_SECONDS`: duration in seconds of a worker wait loop iteration, before checking if resources are available and processing a job if any is available. Defaults to `5`.

For example:

```bash
LOG_LEVEL=DEBUG make datasets-worker
```
