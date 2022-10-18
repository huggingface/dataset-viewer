# libqueue

A Python library to manage the job queues to precompute API responses. The job queues are stored in a mongo database.

## Configuration

Set environment variables to configure the following aspects:

- `QUEUE_MAX_JOBS_PER_DATASET`: the maximum number of started jobs for the same dataset. Defaults to 1.
- `QUEUE_MAX_LOAD_PCT`: the maximum load of the machine (in percentage: the max between the 1m load and the 5m load divided by the number of cpus \*100) allowed to start a job. Set to 0 to disable the test. Defaults to 70.
- `QUEUE_MAX_MEMORY_PCT`: the maximum memory (RAM + SWAP) usage of the machine (in percentage) allowed to start a job. Set to 0 to disable the test. Defaults to 80.
- `QUEUE_MONGO_DATABASE`: the name of the database used for storing the queue. Defaults to `"datasets_server_queue"`.
- `QUEUE_MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.
- `QUEUE_SLEEP_SECONDS`: duration in seconds of a worker wait loop iteration, before checking if resources are available and processing a job if any is available. Note that the worker does not sleep on the first loop after finishing a job. Defaults to `15`.
