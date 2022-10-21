# Datasets server - worker

> Worker that pre-computes and caches the response to /splits

## Configuration

The worker con be configured using environment variables. They are grouped by scope.

### Datasets library

The following environment variables are used to configure two dependencies: the `datasets` and `numba` libraries:

- `HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets data. Defaults to `~/.cache/huggingface/datasets`.
- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached datasets scripts. Defaults to `~/.cache/huggingface/modules`.
- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594).

### Cache

See [../../libs/libcache/README.md](../../libs/libcache/README.md) for more information about the cache configuration.

### Queue

See [../../libs/libqueue/README.md](../../libs/libqueue/README.md) for more information about the queue configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
