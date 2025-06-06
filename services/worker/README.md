# Dataset viewer - worker

> Workers that pre-compute and cache the response for each of the processing steps.

## Configuration

Use environment variables to configure the workers. The prefix of each environment variable gives its scope.

### Uvicorn

The following environment variables are used to configure the Uvicorn server (`WORKER_UVICORN_` prefix). It is used for the /healthcheck and the /metrics endpoints:

- `WORKER_UVICORN_HOSTNAME`: the hostname. Defaults to `"localhost"`.
- `WORKER_UVICORN_NUM_WORKERS`: the number of uvicorn workers. Defaults to `2`.
- `WORKER_UVICORN_PORT`: the port. Defaults to `8000`.

### Prometheus

- `PROMETHEUS_MULTIPROC_DIR`: the directory where the uvicorn workers share their prometheus metrics. See https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn. Defaults to empty, in which case every uvicorn worker manages its own metrics, and the /metrics endpoint returns the metrics of a random worker.

## Worker configuration

Set environment variables to configure the worker.

- `WORKER_CONTENT_MAX_BYTES`: the maximum size in bytes of the response content computed by a worker (to prevent returning big responses in the REST API). Defaults to `10_000_000`.
- `WORKER_DIFFICULTY_MAX`: the maximum difficulty (included) of the jobs to process. Difficulty will always be a strictly positive integer, and its max value is 100. Defaults to None.
- `WORKER_DIFFICULTY_MIN`: the minimum difficulty (excluded) of the jobs to process. Difficulty will always be a strictly positive integer, and its max value is 100. Defaults to None.
- `WORKER_HEARTBEAT_INTERVAL_SECONDS`: the time interval between two heartbeats. Each heartbeat updates the job "last_heartbeat" field in the queue. Defaults to `60` (1 minute).
- `WORKER_KILL_LONG_JOB_INTERVAL_SECONDS`: the time interval at which the worker looks for long jobs to kill them. Defaults to `60` (1 minute).
- `WORKER_KILL_ZOMBIES_INTERVAL_SECONDS`: the time interval at which the worker looks for zombie jobs to kill them. Defaults to `600` (10 minutes).
- `WORKER_MAX_JOB_DURATION_SECONDS`: the maximum duration allowed for a job to run. If the job runs longer, it is killed (see `WORKER_KILL_LONG_JOB_INTERVAL_SECONDS`). Defaults to `1200` (20 minutes).
- `WORKER_MAX_LOAD_PCT`: maximum load of the machine (in percentage: the max between the 1m load and the 5m load divided by the number of CPUs \*100) allowed to start a job. Set to 0 to disable the test. Defaults to 70.
- `WORKER_MAX_MEMORY_PCT`: maximum memory (RAM + SWAP) usage of the machine (in percentage) allowed to start a job. Set to 0 to disable the test. Defaults to 80.
- `WORKER_MAX_MISSING_HEARTBEATS`: the number of hearbeats a job must have missed to be considered a zombie job. Defaults to `5`.
- `WORKER_SLEEP_SECONDS`: wait duration in seconds at each loop iteration before checking if resources are available and processing a job if any is available. Note that the loop doesn't wait just after finishing a job: the next job is immediately processed. Defaults to `15`.

Also, it's possible to force the parent directory in which the temporary files (as the current job state file and its associated lock file) will be created by setting `TMPDIR` to a writable directory. If not set, the worker will use the default temporary directory of the system, as described in https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir.

### Datasets based worker

Set environment variables to configure the datasets-based worker (`DATASETS_BASED_` prefix):

- `DATASETS_BASED_HF_DATASETS_CACHE`: directory where the `datasets` library will store the cached datasets' data. If not set, the datasets library will choose the default location. Defaults to None.

Also, set the modules cache configuration for the datasets-based worker. See [../../libs/libcommon/README.md](../../libs/libcommon/README.md). Note that this variable has no `DATASETS_BASED_` prefix:

- `HF_MODULES_CACHE`: directory where the `datasets` library will store the cached dataset scripts. If not set, the datasets library will choose the default location. Defaults to None.

### Numba library

Numba requires setting the `NUMBA_CACHE_DIR` environment variable to a writable directory to cache the compiled functions. Required on cloud infrastructure (see https://stackoverflow.com/a/63367171/7351594):

- `NUMBA_CACHE_DIR`: directory where the `numba` decorators (used by `librosa`) can write cache.

### Huggingface_hub library

If the Hub is not https://huggingface.co (i.e., if you set the `COMMON_HF_ENDPOINT` environment variable), you must set the `HF_ENDPOINT` environment variable to the same value. See https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191411 for more details:

- `HF_ENDPOINT`: the URL of the Hub. Defaults to `https://huggingface.co`.

### First rows worker

Set environment variables to configure the `first-rows` worker (`FIRST_ROWS_` prefix):

- `FIRST_ROWS_MAX_BYTES`: the max size of the /first-rows response in bytes. Defaults to `1_000_000` (1 MB).
- `FIRST_ROWS_MIN_CELL_BYTES`: the minimum size in bytes of a cell when truncating the content of a row (see `FIRST_ROWS_ROWS_MAX_BYTES`). Below this limit, the cell content will not be truncated. Defaults to `100`.
- `FIRST_ROWS_MIN_NUMBER`: the min number of rows fetched by the worker for the split and provided in the /first-rows response. Defaults to `10`.
- `FIRST_ROWS_COLUMNS_MAX_NUMBER`: the max number of columns (features) provided in the /first-rows response. If the number of columns is greater than the limit, an error is returned. Defaults to `1_000`.

Also, set the assets-related configuration for the first-rows worker. See [../../libs/libcommon/README.md](../../libs/libcommon/README.md).

### Parquet and info worker

Set environment variables to configure the `parquet-and-info` worker (`PARQUET_AND_INFO_` prefix):

- `PARQUET_AND_INFO_COMMIT_MESSAGE`: the git commit message when the worker uploads the parquet files to the Hub. Defaults to `Update parquet files`.
- `COMMITTER_HF_TOKEN`: the HuggingFace token to commit the parquet files to the Hub. The token must be an app token associated with a user that has the right to 1. create the `refs/convert/parquet` branch (see `PARQUET_AND_INFO_TARGET_REVISION`) and 2. push commits to it on any dataset. [Datasets maintainers](https://huggingface.co/datasets-maintainers) members have these rights. The token must have permission to write. If not set, the worker will fail. Defaults to None.
- `PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES`: the maximum size in bytes of the dataset to pre-compute the parquet files. Bigger datasets, or datasets without that information, are partially streamed to get parquet files up to this value. Defaults to `100_000_000`.
- `PARQUET_AND_INFO_MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY`: the maximum size in bytes of the row groups of parquet datasets that are copied to the target revision. Bigger datasets, or datasets without that information, are partially streamed to get parquet files up to `PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES` bytes. Defaults to `100_000_000`.
- `PARQUET_AND_INFO_SOURCE_REVISION`: the git revision of the dataset to use to prepare the parquet files. Defaults to `main`.
- `PARQUET_AND_INFO_TARGET_REVISION`: the git revision of the dataset where to store the parquet files. Make sure the committer token (`PARQUET_AND_INFO_COMMITTER_HF_TOKEN`) has the permission to write there. Defaults to `refs/convert/parquet`.
- `PARQUET_AND_INFO_URL_TEMPLATE`: the URL template to build the parquet file URLs. Defaults to `/datasets/%s/resolve/%s/%s`.

### Descriptive statistics worker

Set environment variables to configure the `descriptive-statistics` worker (`DESCRIPTIVE_STATISTICS_` prefix):

- `DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY`: directory to which a dataset in parquet format is downloaded. Defaults to empty.
- `DESCRIPTIVE_STATISTICS_MAX_SPLIT_SIZE_BYTES`: if size in bytes of raw uncompressed split data is larger than this value, only first `n` parquet files are used so that their sum of uncompressed content in bytes is not greater than approximately `DESCRIPTIVE_STATISTICS_MAX_SPLIT_SIZE_BYTES`. Defaults to `100_000_000`.
- 
#### How descriptive statistics are computed 

Descriptive statistics are currently computed for the following data types: strings, floats, and ints (including `ClassLabel` int). 
The response has three fields: `num_examples`, `statistics`, and `partial`. `partial` indicates if statistics are computed over the first ~`DESCRIPTIVE_STATISTICS_MAX_SPLIT_SIZE_BYTES` of a dataset: `partial: True` means that `num_examples` corresponds to the number of examples in this first chunk of data, not of the entire split. 
`statistics` field is a list of dicts with three keys: `column_name`, `column_type`, and `column_statistics`.

`column_type` is one of the following values:
* `class_label` - for `datasets.ClassLabel` feature
* `float` - for float dtypes ("float16", "float32", "float64")
* `int` - for integer dtypes ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
* `string_label` - for string dtypes ("string", "large_string") - if there are less than or equal to `MAX_NUM_STRING_LABELS` unique values (hardcoded in worker's code, for now it's 30)
* `string_text` - for string dtypes ("string", "large_string") - if there are more than `MAX_NUM_STRING_LABELS` unique values
* `bool` - for boolean dtype ("bool")
* `list` - for lists of other data types (including lists)
* `audio` - for audio data
* `image` - for image data
* `datetime` - for datetime data

`column_statistics` content depends on the feature type, see examples below.
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

Bin size for histogram is counted as `(max_value - min_value) / NUM_BINS`. Currently `NUM_BINS` is 10.

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

As bin edges for integer values also must be integers, bin size is counted as `np.ceil((max_value - min_value + 1) / NUM_BINS)`. Currently `NUM_BINS` is 10. Rounding up means that there might be smaller number of bins in response then `NUM_BINS`. The last bin's size might be smaller than that of the others if the feature's range is not divisible by the rounded bin size. 

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

If the proportion of unique values in a column (within requested split) is <= `MAX_PROPORTION_STRING_LABELS` (currently 0.2) and the number of unique values is <= `MAX_NUM_STRING_LABELS` (currently 1000), the column is considered to be a category and the categories counts are computed. If the proportion on unique values is > `MAX_PROPORTION_STRING_LABELS` but the number of unique values is <= `NUM_BINS`, it is still treated as category.

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

If a string column doesn't satisfy the conditions to be considered a category (see above), it is considered to be text and the distribution of text **lengths** is computed.

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

##### bool

<details><summary>example: </summary>
<p>

```python
{
    'column_name': 'bool__nan_column', 
    'column_type': 'bool', 
    'column_statistics': 
        {
            'nan_count': 3, 
            'nan_proportion': 0.15, 
            'frequencies': {
                'False': 7, 
                'True': 10
            }
        }
}
```
</p>
</details>

##### list

Show distribution of lists lengths. Note: dictionaries of lists are not supported (only lists of dictionaries).

<details><summary>example: </summary>
<p>

```python
{
    "column_name": "list_col",
    "column_type": "list",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 1,
        "max": 3,
        "mean": 1.01741,
        "median": 1.0,
        "std": 0.13146,
        "histogram": {
            "hist": [
                11177,
                196,
                1
            ],
            "bin_edges": [
                1,
                2,
                3,
                3
            ]
        }
    }
}
```
</p>
</details>

##### audio

Shows distribution of audio files durations.

<details><summary>example: </summary>
<p>

```python
{
    "column_name": "audio_col",
    "column_type": "audio",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "min": 1.02,
        "max": 15,
        "mean": 13.93042,
        "median": 14.77,
        "std": 2.63734,
        "histogram": {
            "hist": [
                32,
                25,
                18,
                24,
                22,
                17,
                18,
                19,
                55,
                1770
            ],
            "bin_edges": [
                1.02,
                2.418,
                3.816,
                5.214,
                6.612,
                8.01,
                9.408,
                10.806,
                12.204,
                13.602,
                15
            ]
        }
    }
}
```
</p>
</details>

##### image

Shows distribution of image files widths.

<details><summary>example: </summary>
<p>

```python
{
    "column_name": "image",
    "column_type": "image",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 256,
        "max": 873,
        "mean": 327.99339,
        "median": 341.0,
        "std": 60.07286,
        "histogram": {
            "hist": [
                1734,
                1637,
                1326,
                121,
                10,
                3,
                1,
                3,
                1,
                2
            ],
            "bin_edges": [
                256,
                318,
                380,
                442,
                504,
                566,
                628,
                690,
                752,
                814,
                873
            ]
        }
    }
}
```
</p>
</details>


##### datetime

Shows distribution of datetimes.

<details><summary>example: </summary>
<p>

```python
{
    "column_name": "date",
    "column_type": "datetime",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": "2013-05-18 04:54:11",
        "max": "2013-06-20 10:01:41",
        "mean": "2013-05-27 18:03:39",
        "median": "2013-05-23 11:55:50",
        "std": "11 days, 4:57:32.322450",
        "histogram": {
            "hist": [
                318776,
                393036,
                173904,
                0,
                0,
                0,
                0,
                0,
                0,
                206284
            ],
            "bin_edges": [
                "2013-05-18 04:54:11",
                "2013-05-21 12:36:57",
                "2013-05-24 20:19:43",
                "2013-05-28 04:02:29",
                "2013-05-31 11:45:15",
                "2013-06-03 19:28:01",
                "2013-06-07 03:10:47",
                "2013-06-10 10:53:33",
                "2013-06-13 18:36:19",
                "2013-06-17 02:19:05",
                "2013-06-20 10:01:41"
            ]
        }
    }
}
```
</p>
</details>

### Splits worker

The `splits` worker does not need any additional configuration.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.
