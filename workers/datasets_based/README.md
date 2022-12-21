# Datasets server - worker

> Worker that pre-computes and caches the response to /splits, /first-rows or /parquet.

## Configuration

Use environment variables to configure the worker. The prefix of each environment variable gives its scope.

### Datasets based worker

Set environment variables to configure the datasets-based worker (`DATASETS_BASED_` prefix):

- `DATASETS_BASED_ENDPOINT`: the endpoint on which the worker will work (pre-compute and cache the response). The same worker is used for different endpoints to reuse shared code and dependencies. But at runtime, the worker is assigned only one endpoint. Allowed values: `/splits`, `/first_rows`, and ` /parquet`. Defaults to `/splits`.
- `DATASETS_BASED_HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets' data. If not set, the datasets library will choose the default location. Defaults to None.

Also, set the modules cache configuration for the datasets-based worker. See [../../libs/libcommon/README.md](../../libs/libcommon/README.md). Note that this variable has no `DATASETS_BASED_` prefix:

- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached dataset scripts. If not set, the datasets library will choose the default location. Defaults to None.

### Numba library

Numba requires setting the `NUMBA_CACHE_DIR` environment variable to a writable directory to cache the compiled functions. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594):

- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache.

### Huggingface_hub library

If the Hub is not https://huggingface.co (i.e., if you set the `COMMON_HF_ENDPOINT` environment variable), you must set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191411 for more details:

- `HF_ENDPOINT`: the URL of the Hub. Defaults to `https://huggingface.co`.

### First rows worker

Only needed when the `DATASETS_BASED_ENDPOINT` is set to `/first-rows`.

Set environment variables to configure the first rows worker (`FIRST_ROWS_` prefix):

- `FIRST_ROWS_FALLBACK_MAX_DATASET_SIZE`: the maximum size in bytes of the dataset to fall back into normal mode if streaming fails. Note that it requires to have the size in the info metadata. Set to `0` to disable the fallback. Defaults to `100_000_000`.
- `FIRST_ROWS_MAX_BYTES`: the max size of the /first-rows endpoint response in bytes. Defaults to `1_000_000` (1 MB).
- `FIRST_ROWS_MAX_NUMBER`: the max number of rows fetched by the worker for the split and provided in the /first-rows endpoint response. Defaults to `100`.
- `FIRST_ROWS_MIN_CELL_BYTES`: the minimum size in bytes of a cell when truncating the content of a row (see `FIRST_ROWS_ROWS_MAX_BYTES`). Below this limit, the cell content will not be truncated. Defaults to `100`.
- `FIRST_ROWS_MIN_NUMBER`: the min number of rows fetched by the worker for the split and provided in the /first-rows endpoint response. Defaults to `10`.

Also, set the assets-related configuration for the first-rows worker. See [../../libs/libcommon/README.md](../../libs/libcommon/README.md).

### Parquet worker

Only needed when the `DATASETS_BASED_ENDPOINT` is set to `/parquet`.

Set environment variables to configure the parquet worker (`PARQUET_` prefix):

- `PARQUET_BLOCKED_DATASETS`: comma-separated list of the blocked datasets. If empty, no dataset is blocked. Defaults to empty.
- `PARQUET_COMMIT_MESSAGE`: the git commit message when the worker uploads the parquet files to the Hub. Defaults to `Update parquet files`.
- `PARQUET_COMMITTER_HF_TOKEN`: the user token (https://huggingface.co/settings/tokens) to commit the parquet files to the Hub. The user must be allowed to create the `refs/convert/parquet` branch (see `PARQUET_TARGET_REVISION`) ([Hugging Face organization](https://huggingface.co/huggingface) members have this right). It must also have the right to push to the `refs/convert/parquet` branch ([Datasets maintainers](https://huggingface.co/datasets-maintainers) members have this right). It must have permission to write. If not set, the worker will fail. Defaults to None.
- `PARQUET_MAX_DATASET_SIZE`: the maximum size in bytes of the dataset to pre-compute the parquet files. Bigger datasets, or datasets without that information, are ignored. Defaults to `100_000_000`.
- `PARQUET_SOURCE_REVISION`: the git revision of the dataset to use to prepare the parquet files. Defaults to `main`.
- `PARQUET_SUPPORTED_DATASETS`: comma-separated list of the supported datasets. The worker does not test the size of supported datasets against the maximum dataset size. Defaults to empty.
- `PARQUET_TARGET_REVISION`: the git revision of the dataset where to store the parquet files. Make sure the committer token (`PARQUET_COMMITTER_HF_TOKEN`) has the permission to write there. Defaults to `refs/convert/parquet`.
- `PARQUET_URL_TEMPLATE`: the URL template to build the parquet file URLs. Defaults to `/datasets/%s/resolve/%s/%s`.

### Splits worker

The splits worker does not need any additional configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
