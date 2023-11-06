# Datasets server - worker

> Workers that pre-compute and cache the response to /splits, /first-rows, /parquet, /info and /size.

## Configuration

Use environment variables to configure the workers. The prefix of each environment variable gives its scope.

## Worker configuration

Set environment variables to configure the worker.

- `WORKER_CONTENT_MAX_BYTES`: the maximum size in bytes of the response content computed by a worker (to prevent returning big responses in the REST API). Defaults to `10_000_000`.
- `WORKER_DIFFICULTY_MAX`: the maximum difficulty of the jobs to process. Defaults to None.
- `WORKER_DIFFICULTY_MIN`: the minimum difficulty of the jobs to process. Defaults to None.
- `WORKER_HEARTBEAT_INTERVAL_SECONDS`: the time interval between two heartbeats. Each heartbeat updates the job "last_heartbeat" field in the queue. Defaults to `60` (1 minute).
- `WORKER_JOB_TYPES_BLOCKED`: comma-separated list of job types that will not be processed, e.g. "dataset-config-names,dataset-split-names". If empty, no job type is blocked. Defaults to empty.
- `WORKER_JOB_TYPES_ONLY`: comma-separated list of the non-blocked job types to process, e.g. "dataset-config-names,dataset-split-names". If empty, the worker processes all the non-blocked jobs. Defaults to empty.
- `WORKER_KILL_LONG_JOB_INTERVAL_SECONDS`: the time interval at which the worker looks for long jobs to kill them. Defaults to `60` (1 minute).
- `WORKER_KILL_ZOMBIES_INTERVAL_SECONDS`: the time interval at which the worker looks for zombie jobs to kill them. Defaults to `600` (10 minutes).
- `WORKER_MAX_DISK_USAGE_PCT`: maximum disk usage of every storage disk in the list (in percentage) to allow a job to start. Set to 0 to disable the test. Defaults to 90.
- `WORKER_MAX_JOB_DURATION_SECONDS`: the maximum duration allowed for a job to run. If the job runs longer, it is killed (see `WORKER_KILL_LONG_JOB_INTERVAL_SECONDS`). Defaults to `1200` (20 minutes).
- `WORKER_MAX_LOAD_PCT`: maximum load of the machine (in percentage: the max between the 1m load and the 5m load divided by the number of CPUs \*100) allowed to start a job. Set to 0 to disable the test. Defaults to 70.
- `WORKER_MAX_MEMORY_PCT`: maximum memory (RAM + SWAP) usage of the machine (in percentage) allowed to start a job. Set to 0 to disable the test. Defaults to 80.
- `WORKER_MAX_MISSING_HEARTBEATS`: the number of hearbeats a job must have missed to be considered a zombie job. Defaults to `5`.
- `WORKER_SLEEP_SECONDS`: wait duration in seconds at each loop iteration before checking if resources are available and processing a job if any is available. Note that the loop doesn't wait just after finishing a job: the next job is immediately processed. Defaults to `15`.
- `WORKER_STORAGE_PATHS`: comma-separated list of paths to check for disk usage. Defaults to empty.

Also, it's possible to force the parent directory in which the temporary files (as the current job state file and its associated lock file) will be created by setting `TMPDIR` to a writable directory. If not set, the worker will use the default temporary directory of the system, as described in https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir.

### Datasets based worker

Set environment variables to configure the datasets-based worker (`DATASETS_BASED_` prefix):

- `DATASETS_BASED_HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets' data. If not set, the datasets library will choose the default location. Defaults to None.

Also, set the modules cache configuration for the datasets-based worker. See [../../libs/libcommon/README.md](../../libs/libcommon/README.md). Note that this variable has no `DATASETS_BASED_` prefix:

- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached dataset scripts. If not set, the datasets library will choose the default location. Defaults to None.

Note that both directories will be appended to `WORKER_STORAGE_PATHS` (see [../../libs/libcommon/README.md](../../libs/libcommon/README.md)) to hold the workers when the disk is full.

### Numba library

Numba requires setting the `NUMBA_CACHE_DIR` environment variable to a writable directory to cache the compiled functions. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594):

- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache.

Note that this directory will be appended to `WORKER_STORAGE_PATHS` (see [../../libs/libcommon/README.md](../../libs/libcommon/README.md)) to hold the workers when the disk is full.

### Huggingface_hub library

If the Hub is not https://huggingface.co (i.e., if you set the `COMMON_HF_ENDPOINT` environment variable), you must set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191411 for more details:

- `HF_ENDPOINT`: the URL of the Hub. Defaults to `https://huggingface.co`.

### First rows worker

Set environment variables to configure the `first-rows` worker (`FIRST_ROWS_` prefix):

- `FIRST_ROWS_MAX_BYTES`: the max size of the /first-rows response in bytes. Defaults to `1_000_000` (1 MB).
- `FIRST_ROWS_MAX_NUMBER`: the max number of rows fetched by the worker for the split and provided in the /first-rows response. Defaults to `100`.
- `FIRST_ROWS_MIN_CELL_BYTES`: the minimum size in bytes of a cell when truncating the content of a row (see `FIRST_ROWS_ROWS_MAX_BYTES`). Below this limit, the cell content will not be truncated. Defaults to `100`.
- `FIRST_ROWS_MIN_NUMBER`: the min number of rows fetched by the worker for the split and provided in the /first-rows response. Defaults to `10`.
- `FIRST_ROWS_COLUMNS_MAX_NUMBER`: the max number of columns (features) provided in the /first-rows response. If the number of columns is greater than the limit, an error is returned. Defaults to `1_000`.

Also, set the assets-related configuration for the first-rows worker. See [../../libs/libcommon/README.md](../../libs/libcommon/README.md).

### Parquet and info worker

Set environment variables to configure the `parquet-and-info` worker (`PARQUET_AND_INFO_` prefix):

- `PARQUET_AND_INFO_COMMIT_MESSAGE`: the git commit message when the worker uploads the parquet files to the Hub. Defaults to `Update parquet files`.
- `PARQUET_AND_INFO_COMMITTER_HF_TOKEN`: the HuggingFace token to commit the parquet files to the Hub. The token must be an app token associated with a user that has the right to 1. create the `refs/convert/parquet` branch (see `PARQUET_AND_INFO_TARGET_REVISION`) and 2. push commits to it on any dataset. [Datasets maintainers](https://huggingface.co/datasets-maintainers) members have these rights. The token must have permission to write. If not set, the worker will fail. Defaults to None.
- `PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES`: the maximum size in bytes of the dataset to pre-compute the parquet files. Bigger datasets, or datasets without that information, are partially streamed to get parquet files up to this value. Defaults to `100_000_000`.
- `PARQUET_AND_INFO_MAX_EXTERNAL_DATA_FILES`: the maximum number of external files of the datasets. Bigger datasets, or datasets without that information, are partially streamed to get parquet files up to `PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES` bytes. Defaults to `10_000`.
- `PARQUET_AND_INFO_MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY`: the maximum size in bytes of the row groups of parquet datasets that are copied to the target revision. Bigger datasets, or datasets without that information, are partially streamed to get parquet files up to `PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES` bytes. Defaults to `100_000_000`.
- `PARQUET_AND_INFO_SOURCE_REVISION`: the git revision of the dataset to use to prepare the parquet files. Defaults to `main`.
- `PARQUET_AND_INFO_TARGET_REVISION`: the git revision of the dataset where to store the parquet files. Make sure the committer token (`PARQUET_AND_INFO_COMMITTER_HF_TOKEN`) has the permission to write there. Defaults to `refs/convert/parquet`.
- `PARQUET_AND_INFO_URL_TEMPLATE`: the URL template to build the parquet file URLs. Defaults to `/datasets/%s/resolve/%s/%s`.

### Duckdb Index worker

Set environment variables to configure the `duckdb-index` worker (`DUCKDB_INDEX_` prefix):

- `DUCKDB_INDEX_CACHE_DIRECTORY`: directory where the temporal duckdb index files are stored. Defaults to empty.
- `DUCKDB_INDEX_COMMIT_MESSAGE`: the git commit message when the worker uploads the duckdb index file to the Hub. Defaults to `Update duckdb index file`.
- `DUCKDB_INDEX_COMMITTER_HF_TOKEN`: the HuggingFace token to commit the duckdb index file to the Hub. The token must be an app token associated with a user that has the right to 1. create the `refs/convert/parquet` branch (see `DUCKDB_INDEX_TARGET_REVISION`) and 2. push commits to it on any dataset. [Datasets maintainers](https://huggingface.co/datasets-maintainers) members have these rights. The token must have permission to write. If not set, the worker will fail. Defaults to None.
- `DUCKDB_INDEX_MAX_DATASET_SIZE_BYTES`: the maximum size in bytes of the dataset's parquet files to index. Datasets with bigger size are ignored. Defaults to `100_000_000`.
- `DUCKDB_INDEX_TARGET_REVISION`: the git revision of the dataset where to store the duckdb index file. Make sure the committer token (`DUCKDB_INDEX_COMMITTER_HF_TOKEN`) has the permission to write there. Defaults to `refs/convert/parquet`.
- `DUCKDB_INDEX_URL_TEMPLATE`: the URL template to build the duckdb index file URL. Defaults to `/datasets/%s/resolve/%s/%s`.
- `DUCKDB_INDEX_EXTENSIONS_DIRECTORY`: directory where the duckdb extensions will be downloaded. Defaults to empty.

### Descriptive statistics worker

Set environment variables to configure the `descriptive-statistics` worker (`DESCRIPTIVE_STATISTICS_` prefix):

- `DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY`: directory to which a dataset in parquet format is downloaded. Defaults to empty.
- `DESCRIPTIVE_STATISTICS_HISTOGRAM_NUM_BINS`: number of histogram bins (see examples below for more info).
- `DESCRIPTIVE_STATISTICS_MAX_PARQUET_SIZE_BYTES`: maximum size in bytes of the dataset's parquet files to compute statistics. Datasets with bigger size are ignored. Defaults to `100_000_000`.

#### How descriptive statistics are computed 

Descriptive statistics are currently computed for the following data types: strings, floats, and ints (including `ClassLabel` int). 
Response has two fields: `num_examples` and `statistics`. `statistics` field is a list of dicts with three keys: `column_name`, `column_type`, and `column_statistics`.

`column_type` is one of the following values:
* `class_label` - for `datasets.ClassLabel` feature
* `float` - for float dtypes ("float16", "float32", "float64")
* `int` - for integer dtypes ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
* `string_label` - for string dtypes ("string", "large_string") - if there are less than or equal to `MAX_NUM_STRING_LABELS` unique values (hardcoded in worker's code, for now it's 30)
* `string_text` - for string dtypes ("string", "large_string") - if there are more than `MAX_NUM_STRING_LABELS` unique values

`column_statistics` content depends on the feature type. 
##### class_label

<details><summary>example: </summary>
<p>

```python
{
    "column_name": "class_col",
    "column_type": "class_label",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "no_label_count": 0,  # number of -1 values - special value of the `datasets` lib to encode `no label` 
        "no_label_proportion": 0.0,
        "n_unique": 5,  # number of unique values (excluding `no label` and nan)
        "frequencies": {   # mapping value -> its count
            "this": 19834,
            "are": 20159,
            "random": 20109,
            "words": 20172,
            "test": 19726
        }
    }
}
```
</p>
</details> 

##### float

Bin size for histogram is counted as `(max_value - min_value) / DESCRIPTIVE_STATISTICS_HISTOGRAM_NUM_BINS`

<details><summary>example: </summary>
<p>

```python
{
    "column_name": "delay",
    "column_type": "float",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": -10.206,
        "max": 8.48053,
        "mean": 2.10174,
        "median": 3.4012,
        "std": 3.12487,
        "histogram": {
            "hist": [
                2,
                34,
                256,
                15198,
                9037,
                2342,
                12743,
                45114,
                14904,
                370
            ],
            "bin_edges": [
                -10.206,
                -8.33734,
                -6.46869,
                -4.60004,
                -2.73139,
                -0.86273,
                1.00592,
                2.87457,
                4.74322,
                6.61188,
                8.48053  # includes maximum value, so len is always len(hist) + 1
            ]
        }
    }
}
```
</p>
</details> 

##### int

As bin edges for integer values also must be integers, bin size is counted as `np.ceil((max_value - min_value + 1) / DESCRIPTIVE_STATISTICS_HISTOGRAM_NUM_BINS)`. Rounding up means that there might be smaller number of bins in response then provided `DESCRIPTIVE_STATISTICS_HISTOGRAM_NUM_BINS`. The last bin's size might be smaller than that of the others if the feature's range is not divisible by the rounded bin size. 

<details><summary>examples: </summary>
<p>

```python
{
    "column_name": "direction",
    "column_type": "int",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 0,
        "max": 1,
        "mean": 0.49925,
        "median": 0.0,
        "std": 0.5,
        "histogram": {
            "hist": [
                50075,
                49925
            ],
            "bin_edges": [
                0,
                1,
                1  # if the last value is equal to the last but one, that means that this bin includes only this value
            ]
        }
    }
},
{
    "column_name": "hour",
    "column_type": "int",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 0,
        "max": 23,
        "mean": 13.44402,
        "median": 14.0,
        "std": 5.49455,
        "histogram": {
            "hist": [
                2694,
                2292,
                16785,
                16326,
                16346,
                17809,
                16546,
                11202
            ],
            "bin_edges": [
                0,
                3,
                6,
                9,
                12,
                15,
                18,
                21,
                23
            ]
        }
    }
},
{
    "column_name": "humidity",
    "column_type": "int",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 54,
        "max": 99,
        "mean": 83.89878,
        "median": 85.0,
        "std": 8.65174,
        "histogram": {
            "hist": [
                554,
                1662,
                3823,
                6532,
                12512,
                17536,
                23871,
                20355,
                12896,
                259
            ],
            "bin_edges": [
                54,
                59,
                64,
                69,
                74,
                79,
                84,
                89,
                94,
                99,
                99
            ]
        }
    }
},
{
    "column_name": "weekday",
    "column_type": "int",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 0,
        "max": 6,
        "mean": 3.08063,
        "median": 3.0,
        "std": 1.90347,
        "histogram": {
            "hist": [
                10282,
                15416,
                15291,
                15201,
                15586,
                15226,
                12998
            ],
            "bin_edges": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                6
            ]
        }
    }
}
```

</p>
</details>

##### string_label

If the number of unique values in a column (within requested split) is <= `MAX_NUM_STRING_LABELS` (currently 30), the column is considered to be a category and the categories counts are computed.

<details><summary>examples: </summary>
<p>

```python
{
    'column_name': 'string_col',
    'column_type': 'string_label',
    'column_statistics': 
        {
            "nan_count": 0,
            "nan_proportion": 0.0,
            "n_unique": 5,  # number of unique values (excluding nan)
            "frequencies": {   # mapping value -> its count
                "this": 19834,
                "are": 20159,
                "random": 20109,
                "words": 20172,
                "test": 19726
        }
    }
}
```
</p>
</details>

##### string_text

If the number of unique values in a column (within requested split) is > `MAX_NUM_STRING_LABELS` (currently 30), the column is considered to be text and the distribution of text **lengths** is computed.

<details><summary>example: </summary>
<p>

```python
{
    'column_name': 'text_col',
    'column_type': 'string_text',
    'column_statistics': {
        'max': 296,
        'mean': 97.46649,
        'median': 88.0,
        'min': 11,
        'nan_count': 0,
        'nan_proportion': 0.0,
        'std': 55.82714,
        'histogram': {
            'bin_edges': [
                11,
                40,
                69,
                98,
                127,
                156,
                185,
                214,
                243,
                272,
                296
            ],
            'hist': [
                171,
                224,
                235,
                180,
                102,
                99,
                53,
                28,
                10,
                2
               ]
             },
    }
}
```
</p>
</details>


### Splits worker

The `splits` worker does not need any additional configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
