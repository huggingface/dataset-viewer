# libcommon

A Python library with common code (cache, queue, workers logic, processing steps, configuration, utils, logging, exceptions) used by the services and the workers

## Assets configuration

Set the assets (images and audio files) environment variables to configure the following aspects:

- `ASSETS_BASE_URL`: base URL for the assets files. Set accordingly to the datasets-server domain, e.g., https://datasets-server.huggingface.co/assets. Defaults to `http://localhost/assets`.
- `ASSETS_STORAGE_PROTOCOL`: fsspec protocol for storage, it can take values `file` or `s3`. Defaults to `file`, which means local file system is used.
- `ASSETS_STORAGE_ROOT`: name of the folder where assets are stored. If using `s3` protocol, the first part of the path is the bucket name. Defaults to `/storage/assets`.

## Cached Assets configuration

Set the cached-assets (images and audio files) environment variables to configure the following aspects:

- `CACHED_ASSETS_BASE_URL`: base URL for the cached assets files. Set accordingly to the datasets-server domain, e.g., https://datasets-server.huggingface.co/cached-assets. Defaults to `http://localhost/cached-assets`.
- `CACHED_ASSETS_STORAGE_PROTOCOL`: fsspec protocol for storage, it can take values `file` or `s3`. Defaults to `file`, which means local file system is used.
- `CACHED_ASSETS_STORAGE_ROOT`: name of the folder where cached assets are stored. If using `s3` protocol, the first part of the path is the bucket name. Defaults to `/storage/cached-assets`.

## CloudFront configuration

Set the CloudFront environment variables to configure the following aspects:

- `CLOUDFRONT_EXPIRATION_SECONDS`: CloudFront expiration delay in seconds. Defaults to `3600`.
- `CLOUDFRONT_KEY_PAIR_ID`: CloudFront key pair ID. Defaults to empty.
- `CLOUDFRONT_PRIVATE_KEY`: CloudFront private key in PEM format. Defaults to empty.

To enable signed URLs on the StorageClient, `CLOUDFRONT_KEY_PAIR_ID` and `CLOUDFRONT_PRIVATE_KEY` must be set, and the StorageClient protocol must be `s3`.

## Common configuration

Set the common environment variables to configure the following aspects:

- `COMMON_BLOCKED_DATASETS`: comma-separated list of the blocked datasets. Unix shell-style wildcards also work in the dataset name for namespaced datasets, for example `some_namespace/*` to block all the datasets in the `some_namespace` namespace. If empty, no dataset is blocked. Defaults to empty.
- `COMMON_DATASET_SCRIPTS_ALLOW_LIST`: comma-separated list of the datasets for which we support dataset scripts. Unix shell-style wildcards also work in the dataset name for namespaced datasets, for example `some_namespace/*` to refer to all the datasets in the `some_namespace` namespace. The keyword `{{ALL_DATASETS_WITH_NO_NAMESPACE}}` refers to all the datasets without namespace. If empty, no dataset with script is supported. Defaults to empty.
- `COMMON_HF_ENDPOINT`: URL of the HuggingFace Hub. Defaults to `https://huggingface.co`.
- `COMMON_HF_TOKEN`: App Access Token (ask moonlanding administrators to get one, only the `read` role is required) to access the gated datasets. Defaults to empty.

## Logs configuration

Set the common environment variables to configure the logs:

- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. Defaults to `INFO`.

## Cache configuration

Set environment variables to configure the storage of precomputed API responses in a MongoDB database (the "cache"):

- `CACHE_MONGO_DATABASE`: name of the database used for storing the cache. Defaults to `datasets_server_cache`.
- `CACHE_MONGO_URL`: URL used to connect to the MongoDB server. Defaults to `mongodb://localhost:27017`.

## Queue configuration

Set environment variables to configure the job queues to precompute API responses. The job queues are stored in a MongoDB database.

- `QUEUE_MONGO_DATABASE`: name of the database used for storing the queue. Defaults to `datasets_server_queue`.
- `QUEUE_MONGO_URL`: URL used to connect to the MongoDB server. Defaults to `mongodb://localhost:27017`.

## S3 configuration

Set environment variables to configure the connection to S3.

- `S3_REGION_NAME`: bucket region name when using `s3` as storage protocol for assets or cached assets. Defaults to `us-east-1`.
- `S3_ACCESS_KEY_ID`: unique identifier associated with an AWS account. It's used to identify the AWS account that is making requests to S3. Defaults to empty.
- `S3_SECRET_ACCESS_KEY`: secret key associated with an AWS account. Defaults to empty.

## Parquet Metadata

- `PARQUET_METADATA_STORAGE_DIRECTORY`: storage directory where parquet metadata files are stored. See https://arrow.apache.org/docs/python/generated/pyarrow.parquet.FileMetaData.html for more information. 

## Rows Index

- `ROWS_INDEX_MAX_ARROW_DATA_IN_MEMORY`: The maximum number of row groups to be loaded in memory.
