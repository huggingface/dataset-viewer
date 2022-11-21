# Datasets server - worker

> Worker that create parquet files and send them to the Hub

## Configuration

The worker can be configured using environment variables. They are grouped by scope.

### Parquet worker

Set environment variables to configure the parquet worker (`PARQUET_` prefix):

- `COMMIT_MESSAGE`: the git commit message when the parquet files are uploaded to the Hub. Defaults to `Update parquet files`.
- `SOURCE_REVISION`: the git revision of the dataset to use to prepare the parquet files. Defaults to `main`.
- `TARGET_REVISION`: the git revision of the dataset where to store the parquet files. Make sure the hf_token (see the "Common" section) allows to write there. Defaults to `refs/convert/parquet`.
- `URL_TEMPLATE`: the URL template to build the parquet file URLs. Defaults to `/datasets/{repo_id}/resolve/{revision}/{filename}`.

### Datasets library

The following environment variables are used to configure two dependencies: the `datasets` and `numba` libraries:

- `HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached dataset's data. Be sure to provide sufficient storage for the downloaded data files and the local copy of the parquet files. Defaults to `~/.cache/huggingface/datasets`.
- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached datasets scripts. Defaults to `~/.cache/huggingface/modules`.
- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594).

If the Hub is not https://huggingface.co (i.e. if you set the `COMMON_HF_ENDPOINT` environment variable), you should also set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196 for more details.

### Cache

See [../../libs/libcache/README.md](../../libs/libcache/README.md) for more information about the cache configuration.

### Queue

See [../../libs/libqueue/README.md](../../libs/libqueue/README.md) for more information about the queue configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
