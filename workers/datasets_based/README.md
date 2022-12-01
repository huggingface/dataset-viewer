# Datasets server - worker

> Worker that pre-computes and caches the response to /splits

## Configuration

The worker can be configured using environment variables. They are grouped by scope.

### Datasets based worker

The same worker is used for different endpoints to reuse shared code and dependencies. But at runtime, the worker is assigned only one endpoint. The endpoint is configured using the `DATASETS_BASED_ENDPOINT` environment variable:

- `DATASETS_BASED_ENDPOINT`: the endpoint on which the worker will work (pre-compute and cache the response). It can only be `/splits` at the moment.

### Datasets library

The following environment variables are used to configure two dependencies: the `datasets` and `numba` libraries:

- `HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets data. Defaults to `~/.cache/huggingface/datasets`.
- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached datasets scripts. Defaults to `~/.cache/huggingface/modules`.
- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594).

If the Hub is not https://huggingface.co (i.e. if you set the `COMMON_HF_ENDPOINT` environment variable), you should also set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196 for more details.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
