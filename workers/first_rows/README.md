# Datasets server - first_rows

> Worker that pre-computes and caches the response to /first-rows

## Configuration

The worker can be configured using environment variables. They are grouped by scope.

### First rows worker

Set environment variables to configure the first rows worker (`FIRST_ROWS_` prefix):

- `FIRST_ROWS_FALLBACK_MAX_DATASET_SIZE`: the maximum size in bytes of the dataset to fallback in normal mode if streaming fails. Note that it requires to have the size in the info metadata. Set to `0` to disable the fallback. Defaults to `100_000_000`.
- `FIRST_ROWS_MAX_BYTES`: the max size of the /first-rows endpoint response in bytes. Defaults to `1_000_000` (1 MB).
- `FIRST_ROWS_MAX_NUMBER`: the max number of rows fetched by the worker for the split, and provided in the /first-rows endpoint response. Defaults to `100`.
- `FIRST_ROWS_MIN_CELL_BYTES`: the minimum size in bytes of a cell when truncating the content of a row (see `FIRST_ROWS_ROWS_MAX_BYTES`). Below this limit, the cell content will not be truncated. Defaults to `100`.
- `FIRST_ROWS_MIN_NUMBER`: the min number of rows fetched by the worker for the split, and provided in the /first-rows endpoint response. Defaults to `10`.

### Datasets library

The following environment variables are used to configure two dependencies: the `datasets` and `numba` libraries:

- `HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets data. Defaults to `~/.cache/huggingface/datasets`.
- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached datasets scripts. Defaults to `~/.cache/huggingface/modules`.
- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594).

If the Hub is not https://huggingface.co (i.e. if you set the `COMMON_HF_ENDPOINT` environment variable), you should also set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196 for more details.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
