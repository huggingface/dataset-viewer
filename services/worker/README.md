# Datasets server - worker

> Worker that pre-computes and caches the response to /splits, /first-rows or /parquet-and-dataset-info.

## Configuration

Use environment variables to configure the worker. The prefix of each environment variable gives its scope.

## Worker configuration

Set environment variables to configure the worker.

- `WORKER_CONTENT_MAX_BYTES`: the maximum size in bytes of the response content computed by a worker (to prevent returning big responses in the REST API). Defaults to `10_000_000`.
- `WORKER_HEARTBEAT_INTERVAL_SECONDS`: the time interval between two heartbeats. Each heartbeat updates the job "last_heartbeat" field in the queue. Defaults to `60` (1 minute).
- `WORKER_KILL_ZOMBIES_INTERVAL_SECONDS`: the time interval at which the worker looks for zombie jobs to kill them. Defaults to `600` (10 minutes).
- `WORKER_MAX_DISK_USAGE_PCT`: maximum disk usage of every storage disk in the list (in percentage) to allow a job to start. Set to 0 to disable the test. Defaults to 90.
- `WORKER_MAX_LOAD_PCT`: maximum load of the machine (in percentage: the max between the 1m load and the 5m load divided by the number of CPUs \*100) allowed to start a job. Set to 0 to disable the test. Defaults to 70.
- `WORKER_MAX_MEMORY_PCT`: maximum memory (RAM + SWAP) usage of the machine (in percentage) allowed to start a job. Set to 0 to disable the test. Defaults to 80.
- `WORKER_MAX_MISSING_HEARTBEATS`: the number of hearbeats a job must have missed to be considered a zombie job. Defaults to `5`.
- `WORKER_ONLY_JOB_TYPES`: comma-separated list of the job types to process, e.g. "/splits,/first-rows". If empty, the worker processes all the jobs. Defaults to empty.
- `WORKER_SLEEP_SECONDS`: wait duration in seconds at each loop iteration before checking if resources are available and processing a job if any is available. Note that the loop doesn't wait just after finishing a job: the next job is immediately processed. Defaults to `15`.
- `WORKER_STORAGE_PATHS`: comma-separated list of paths to check for disk usage. Defaults to empty.

Also, it's possible to force the parent directory in which the temporary files (as the current job state file and its associated lock file) will be created by setting `TMPDIR` to a writable directory. If not set, the worker will use the default temporary directory of the system, as described in https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir.

### Datasets based worker

Set environment variables to configure the datasets-based worker (`DATASETS_BASED_` prefix):

- `DATASETS_BASED_HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets' data. If not set, the datasets library will choose the default location. Defaults to None.

Also, set the modules cache configuration for the datasets-based worker. See [../../libs/libcommon/README.md](../../libs/libcommon/README.md). Note that this variable has no `DATASETS_BASED_` prefix:

- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached dataset scripts. If not set, the datasets library will choose the default location. Defaults to None.

Note that both directories will be appended to `WORKER_STORAGE_PATHS` (see [../../libs/libcommon/README.md](../../libs/libcommon/README.md)) to hold the workers when the disk is full.

### Numba library

Numba requires setting the `NUMBA_CACHE_DIR` environment variable to a writable directory to cache the compiled functions. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594):

- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache.

Note that this directory will be appended to `WORKER_STORAGE_PATHS` (see [../../libs/libcommon/README.md](../../libs/libcommon/README.md)) to hold the workers when the disk is full.

### Huggingface_hub library

If the Hub is not https://huggingface.co (i.e., if you set the `COMMON_HF_ENDPOINT` environment variable), you must set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191411 for more details:

- `HF_ENDPOINT`: the URL of the Hub. Defaults to `https://huggingface.co`.

### First rows worker

Set environment variables to configure the first rows worker (`FIRST_ROWS_` prefix):

- `FIRST_ROWS_MAX_BYTES`: the max size of the /first-rows response in bytes. Defaults to `1_000_000` (1 MB).
- `FIRST_ROWS_MAX_NUMBER`: the max number of rows fetched by the worker for the split and provided in the /first-rows response. Defaults to `100`.
- `FIRST_ROWS_MIN_CELL_BYTES`: the minimum size in bytes of a cell when truncating the content of a row (see `FIRST_ROWS_ROWS_MAX_BYTES`). Below this limit, the cell content will not be truncated. Defaults to `100`.
- `FIRST_ROWS_MIN_NUMBER`: the min number of rows fetched by the worker for the split and provided in the /first-rows response. Defaults to `10`.
- `FIRST_ROWS_COLUMNS_MAX_NUMBER`: the max number of columns (features) provided in the /first-rows response. If the number of columns is greater than the limit, an error is returned. Defaults to `1_000`.

Also, set the assets-related configuration for the first-rows worker. See [../../libs/libcommon/README.md](../../libs/libcommon/README.md).

### Parquet and dataset info worker

Set environment variables to configure the parquet worker (`PARQUET_AND_DATASET_INFO_` prefix):

- `PARQUET_AND_DATASET_INFO_BLOCKED_DATASETS`: comma-separated list of the blocked datasets. If empty, no dataset is blocked. Defaults to empty.
- `PARQUET_AND_DATASET_INFO_COMMIT_MESSAGE`: the git commit message when the worker uploads the parquet files to the Hub. Defaults to `Update parquet files`.
- `PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN`: the user token (https://huggingface.co/settings/tokens) to commit the parquet files to the Hub. The user must be allowed to create the `refs/convert/parquet` branch (see `PARQUET_AND_DATASET_INFO_TARGET_REVISION`) ([Hugging Face organization](https://huggingface.co/huggingface) members have this right). It must also have the right to push to the `refs/convert/parquet` branch ([Datasets maintainers](https://huggingface.co/datasets-maintainers) members have this right). It must have permission to write. If not set, the worker will fail. Defaults to None.
- `PARQUET_AND_DATASET_INFO_MAX_DATASET_SIZE`: the maximum size in bytes of the dataset to pre-compute the parquet files. Bigger datasets, or datasets without that information, are ignored. Defaults to `100_000_000`.
- `PARQUET_AND_DATASET_INFO_SOURCE_REVISION`: the git revision of the dataset to use to prepare the parquet files. Defaults to `main`.
- `PARQUET_AND_DATASET_INFO_SUPPORTED_DATASETS`: comma-separated list of the supported datasets. The worker does not test the size of supported datasets against the maximum dataset size. Defaults to empty.
- `PARQUET_AND_DATASET_INFO_TARGET_REVISION`: the git revision of the dataset where to store the parquet files. Make sure the committer token (`PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN`) has the permission to write there. Defaults to `refs/convert/parquet`.
- `PARQUET_AND_DATASET_INFO_URL_TEMPLATE`: the URL template to build the parquet file URLs. Defaults to `/datasets/%s/resolve/%s/%s`.

### Splits worker

The splits worker does not need any additional configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
