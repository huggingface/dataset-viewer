# Datasets server - worker

> Worker to pre-process datasets and splits

## Install

See [INSTALL](./INSTALL.md#Install)

## Run

Launch the worker to preprocess the datasets queue:

```bash
WORKER_QUEUE=datasets make run
```

Launch the worker to preprocess the splits queue:

```bash
WORKER_QUEUE=splits make run
```

Launch the worker to preprocess the splits-next/ responses:

```bash
WORKER_QUEUE=splits_responses make run
```

Launch the worker to preprocess the first-rows/ responses:

```bash
WORKER_QUEUE=first_rows_responses make run
```

Set environment variables to configure the following aspects:

- `ASSETS_BASE_URL`: base URL for the assets files. It should be set accordingly to the datasets-server domain, eg https://datasets-server.huggingface.co/assets. Defaults to `assets`.
- `ASSETS_DIRECTORY`: directory where the asset files are stored. Defaults to empty, in which case the assets are located in the `datasets_server_assets` subdirectory inside the OS default cache directory.
- `DATASETS_REVISION`: git reference for the canonical datasets on https://github.com/huggingface/datasets. Defaults to `main`.
- `HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets data. Defaults to `~/.cache/huggingface/datasets`.
- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached datasets scripts. Defaults to `~/.cache/huggingface/modules`.
- `HF_ENDPOINT`: URL of the HuggingFace Hub. Defaults to "https://huggingface.co".
- `HF_TOKEN`: App Access Token (ask moonlanding administrators to get one, only the `read` role is required), to access the gated datasets. Defaults to empty.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_JOB_RETRIES`: the maximum number of job retries (for uncaught errors, such as RAM shortage) for the same job. The job is re-enqueued if an unexpected server error occurred and if its "retries" number is under `MAX_JOB_RETRIES`. Defaults to 3.
- `MAX_JOBS_PER_DATASET`: the maximum number of started jobs for the same dataset. Defaults to 1.
- `MAX_LOAD_PCT`: the maximum load of the machine (in percentage: the max between the 1m load and the 5m load divided by the number of cpus \*100) allowed to start a job. Set to 0 to disable the test. Defaults to 70.
- `MAX_MEMORY_PCT`: the maximum memory (RAM + SWAP) usage of the machine (in percentage) allowed to start a job. Set to 0 to disable the test. Defaults to 80.
- `MAX_SIZE_FALLBACK`: the maximum size in bytes of the dataset to fallback in normal mode if streaming fails. Note that it requires to have the size in the info metadata. Set to `0` to disable the fallback. Defaults to `100_000_000`.
- `MIN_CELL_BYTES`: the minimum size in bytes of a cell when truncating the content of a row (see `ROWS_MAX_BYTES`). Below this limit, the cell content will not be truncated. Defaults to `100`.
- `MONGO_CACHE_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_server_cache"`.
- `MONGO_QUEUE_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_server_queue"`.
- `MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.
- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594).
- `ROWS_MAX_BYTES`: the max size of the /rows endpoint response in bytes. Defaults to `1_000_000` (1 MB).
- `ROWS_MAX_NUMBER`: the max number of rows fetched by the worker for the split, and provided in the /rows endpoint response. Defaults to `100`.
- `ROWS_MIN_NUMBER`: the min number of rows fetched by the worker for the split, and provided in the /rows endpoint response. Defaults to `10`.
- `WORKER_QUEUE`: name of the queue the worker will pull jobs from. It can be equal to `datasets`, `splits`, `splits_responses` or `first_rows_responses`. The `datasets` and `splits_responses` jobs should be a lot faster than the `splits` or `first_rows_responses` ones, so that we should need a lot more workers for `splits`/`first_rows_responses` than for `datasets`/`splits_responses`. Defaults to `datasets`.
- `WORKER_SLEEP_SECONDS`: duration in seconds of a worker wait loop iteration, before checking if resources are available and processing a job if any is available. Note that the worker does not sleep on the first loop after finishing a job. Defaults to `15`.

For example:

```bash
LOG_LEVEL=DEBUG WORKER_QUEUE=datasets make run
```
