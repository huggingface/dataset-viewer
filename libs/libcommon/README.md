# libcommon

A Python library with common code (configuration, utils, logging, exceptions) used by the services and the workers

## Configuration

Set environment variables to configure the following aspects:

- `COMMON_ASSETS_BASE_URL`: base URL for the assets files. It should be set accordingly to the datasets-server domain, eg https://datasets-server.huggingface.co/assets. Defaults to `assets`.
- `COMMON_HF_ENDPOINT`: URL of the HuggingFace Hub. Defaults to `https://huggingface.co`.
- `COMMON_HF_TOKEN`: App Access Token (ask moonlanding administrators to get one, only the `read` role is required), to access the gated datasets. Defaults to empty.
- `COMMON_LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
