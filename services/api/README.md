# Dataset viewer API

> API for HuggingFace 🤗 dataset viewer

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### API service

See [../../libs/libapi/README.md](../../libs/libapi/README.md) for more information about the API configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

See https://huggingface.co/docs/dataset-viewer

- /healthcheck: Ensure the app is running
- /metrics: Return a list of metrics in the Prometheus format
- /croissant-crumbs: Return (parts of) the [Croissant](https://huggingface.co/docs/dataset-viewer/croissant) metadata for a dataset.
- /is-valid: Tell if a dataset is [valid](https://huggingface.co/docs/dataset-viewer/valid)
- /splits: List the [splits](https://huggingface.co/docs/dataset-viewer/splits) names for a dataset
- /first-rows: Extract the [first rows](https://huggingface.co/docs/dataset-viewer/first_rows) for a dataset split
- /parquet: List the [parquet files](https://huggingface.co/docs/dataset-viewer/parquet) auto-converted for a dataset
- /opt-in-out-urls: Return the number of opted-in/out image URLs. See [Spawning AI](https://api.spawning.ai/spawning-api) for more information.
- /statistics: Return some basic statistics for a dataset split.
