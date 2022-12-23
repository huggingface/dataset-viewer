# Datasets server - worker

> Worker that pre-computes and caches the response to /size.

## Configuration

Use environment variables to configure the worker. The prefix of each environment variable gives its scope.

### Parquet-based worker

Set environment variables to configure the datasets-based worker (`PARQUET_BASED_` prefix):

- `PARQUET_BASED_ENDPOINT`: the endpoint on which the worker will work (pre-compute and cache the response). The same worker is used for different endpoints to reuse shared code and dependencies. But at runtime, the worker is assigned only one endpoint. Allowed values: `/size`. Defaults to `/size`.

### Huggingface_hub library

If the Hub is not https://huggingface.co (i.e., if you set the `COMMON_HF_ENDPOINT` environment variable), you must set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191411 for more details:

- `HF_ENDPOINT`: the URL of the Hub. Defaults to `https://huggingface.co`.

### Size worker

The /size worker does not need any additional configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
