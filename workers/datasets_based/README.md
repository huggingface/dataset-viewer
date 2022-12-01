# Datasets server - worker

> Worker that pre-computes and caches the response to /splits, /first-rows or /parquet.

## Configuration

The worker can be configured using environment variables. They are grouped by scope.

### Datasets based worker

The same worker is used for different endpoints to reuse shared code and dependencies. But at runtime, the worker is assigned only one endpoint. The endpoint is configured using the `DATASETS_BASED_ENDPOINT` environment variable:

- `DATASETS_BASED_ENDPOINT`: the endpoint on which the worker will work (pre-compute and cache the response). It can only be `/splits` at the moment.

### First rows worker

Only needed when the `DATASETS_BASED_ENDPOINT` is set to `/first-rows`: set environment variables to configure the first rows worker (`FIRST_ROWS_` prefix):

- `FIRST_ROWS_FALLBACK_MAX_DATASET_SIZE`: the maximum size in bytes of the dataset to fallback in normal mode if streaming fails. Note that it requires to have the size in the info metadata. Set to `0` to disable the fallback. Defaults to `100_000_000`.
- `FIRST_ROWS_MAX_BYTES`: the max size of the /first-rows endpoint response in bytes. Defaults to `1_000_000` (1 MB).
- `FIRST_ROWS_MAX_NUMBER`: the max number of rows fetched by the worker for the split, and provided in the /first-rows endpoint response. Defaults to `100`.
- `FIRST_ROWS_MIN_CELL_BYTES`: the minimum size in bytes of a cell when truncating the content of a row (see `FIRST_ROWS_ROWS_MAX_BYTES`). Below this limit, the cell content will not be truncated. Defaults to `100`.
- `FIRST_ROWS_MIN_NUMBER`: the min number of rows fetched by the worker for the split, and provided in the /first-rows endpoint response. Defaults to `10`.

### Parquet worker

Only needed when the `DATASETS_BASED_ENDPOINT` is set to `/parquet`: set environment variables to configure the parquet worker (`PARQUET_` prefix):

- `COMMIT_MESSAGE`: the git commit message when the parquet files are uploaded to the Hub. Defaults to `Update parquet files`.
- `SOURCE_REVISION`: the git revision of the dataset to use to prepare the parquet files. Defaults to `main`.
- `SUPPORTED_DATASETS`: comma-separated list of the supported datasets. If empty, all the datasets are processed. Defaults to empty.
- `TARGET_REVISION`: the git revision of the dataset where to store the parquet files. Make sure the hf_token (see the "Common" section) allows to write there. Defaults to `refs/convert/parquet`.
- `URL_TEMPLATE`: the URL template to build the parquet file URLs. Defaults to `/datasets/%s/resolve/%s/%s`.

### Datasets library

The following environment variables are used to configure two dependencies: the `datasets` and `numba` libraries:

- `HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets data. Defaults to `~/.cache/huggingface/datasets`.
- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached datasets scripts. Defaults to `~/.cache/huggingface/modules`.
- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594).

If the Hub is not https://huggingface.co (i.e. if you set the `COMMON_HF_ENDPOINT` environment variable), you should also set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196 for more details.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
