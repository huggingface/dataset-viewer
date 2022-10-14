# Datasets server - worker

> Worker to pre-process datasets and splits

## Configuration

Set environment variables to configure the following aspects:

- `DATASETS_REVISION`: git reference for the canonical datasets on https://github.com/huggingface/datasets. Defaults to `main`.
- `HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets data. Defaults to `~/.cache/huggingface/datasets`.
- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached datasets scripts. Defaults to `~/.cache/huggingface/modules`.
- `HF_ENDPOINT`: URL of the HuggingFace Hub. Defaults to `https://huggingface.co`.
- `HF_TOKEN`: App Access Token (ask moonlanding administrators to get one, only the `read` role is required), to access the gated datasets. Defaults to empty.
- `LOG_LEVEL`: log level, among `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. Defaults to `INFO`.
- `MAX_JOBS_PER_DATASET`: the maximum number of started jobs for the same dataset. Defaults to 1.
- `MAX_LOAD_PCT`: the maximum load of the machine (in percentage: the max between the 1m load and the 5m load divided by the number of cpus \*100) allowed to start a job. Set to 0 to disable the test. Defaults to 70.
- `MAX_MEMORY_PCT`: the maximum memory (RAM + SWAP) usage of the machine (in percentage) allowed to start a job. Set to 0 to disable the test. Defaults to 80.
- `MONGO_CACHE_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_server_cache"`.
- `MONGO_QUEUE_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_server_queue"`.
- `MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.
- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594).
- `WORKER_SLEEP_SECONDS`: duration in seconds of a worker wait loop iteration, before checking if resources are available and processing a job if any is available. Note that the worker does not sleep on the first loop after finishing a job. Defaults to `15`.
