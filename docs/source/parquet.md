# List Parquet files

Datasets can be published in any format (CSV, JSONL, directories of images, etc.) to the Hub, and they are easily accessed with the 🤗 [Datasets](https://huggingface.co/docs/datasets/) library. For a more performant experience (especially when it comes to large datasets), the dataset viewer automatically converts every dataset to the [Parquet](https://parquet.apache.org/) format.

## What is Parquet?

Parquet is a columnar storage format optimized for querying and processing large datasets. Parquet is a popular choice for big data processing and analytics and is widely used for data processing and machine learning.

In Parquet, data is divided into chunks called "row groups", and within each row group, it is stored in columns rather than rows. Each row group column is compressed separately using the best compression algorithm depending on the data, and contains metadata and statistics (min/max value, number of NULL values) about the data it contains.

This structure allows for efficient data reading and querying:
- only the necessary columns are read from disk (projection pushdown); no need to read the entire file. This reduces the memory requirement for working with Parquet data. 
- entire row groups are skipped if the statistics stored in its metadata do not match the data of interest (automatic filtering)
- the data is compressed, which reduces the amount of data that needs to be stored and transferred.

A Parquet file contains a single table. If a dataset has multiple tables (e.g. multiple splits or configurations), each table is stored in a separate Parquet file.

## Conversion to Parquet

The Parquet files are published to the Hub on a specific `refs/convert/parquet` branch (like this `amazon_polarity` [branch](https://huggingface.co/datasets/amazon_polarity/tree/refs%2Fconvert%2Fparquet) for example) that parallels the `main` branch.

<Tip>

In order for the dataset viewer to generate a Parquet version of a dataset, the dataset must be _public_, or owned by a [PRO user](https://huggingface.co/pricing) or an [Enterprise Hub organization](https://huggingface.co/enterprise).

</Tip>

## Using the dataset viewer API

This guide shows you how to use the dataset viewer's `/parquet` endpoint to retrieve a list of a dataset's files converted to Parquet. Feel free to also try it out with [Postman](https://www.postman.com/huggingface/workspace/hugging-face-apis/request/23242779-f0cde3b9-c2ee-4062-aaca-65c4cfdd96f8), [RapidAPI](https://rapidapi.com/hugging-face-hugging-face-default/api/hugging-face-datasets-api), or [ReDoc](https://redocly.github.io/redoc/?url=https://datasets-server.huggingface.co/openapi.json#operation/listSplits).

The `/parquet` endpoint accepts the dataset name as its query parameter:

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/parquet?dataset=ibm/duorc"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
```
</python>
<js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://datasets-server.huggingface.co/parquet?dataset=ibm/duorc",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const result = await response.json();
    return result;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js>
<curl>
```curl
curl https://datasets-server.huggingface.co/parquet?dataset=ibm/duorc \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The endpoint response is a JSON containing a list of the dataset's files in the Parquet format. For example, the [`ibm/duorc`](https://huggingface.co/datasets/ibm/duorc) dataset has six Parquet files, which corresponds to the `test`, `train` and `validation` splits of its two configurations, `ParaphraseRC` and `SelfRC` (see the [List splits and configurations](./splits) guide for more details about splits and configurations).

The endpoint also gives the filename and size of each file:

```json
{
   "parquet_files":[
      {
         "dataset":"ibm/duorc",
         "config":"ParaphraseRC",
         "split":"test",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/ParaphraseRC/test/0000.parquet",
         "filename":"0000.parquet",
         "size":6136591
      },
      {
         "dataset":"ibm/duorc",
         "config":"ParaphraseRC",
         "split":"train",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/ParaphraseRC/train/0000.parquet",
         "filename":"0000.parquet",
         "size":26005668
      },
      {
         "dataset":"ibm/duorc",
         "config":"ParaphraseRC",
         "split":"validation",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/ParaphraseRC/validation/0000.parquet",
         "filename":"0000.parquet",
         "size":5566868
      },
      {
         "dataset":"ibm/duorc",
         "config":"SelfRC",
         "split":"test",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/SelfRC/test/0000.parquet",
         "filename":"0000.parquet",
         "size":3035736
      },
      {
         "dataset":"ibm/duorc",
         "config":"SelfRC",
         "split":"train",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/SelfRC/train/0000.parquet",
         "filename":"0000.parquet",
         "size":14851720
      },
      {
         "dataset":"ibm/duorc",
         "config":"SelfRC",
         "split":"validation",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/SelfRC/validation/0000.parquet",
         "filename":"0000.parquet",
         "size":3114390
      }
   ],
   "pending":[
      
   ],
   "failed":[
      
   ],
   "partial":false
}
```

## Sharded Parquet files

Big datasets are partitioned into Parquet files (shards) of about 500MB each. The filename contains the name of the dataset, the split, the shard index, and the total number of shards (`dataset-name-train-0000-of-0004.parquet`). For a given split, the elements in the list are sorted by their shard index, in ascending order. For example, the `train` split of the [`amazon_polarity`](https://datasets-server.huggingface.co/parquet?dataset=amazon_polarity) dataset is partitioned into 4 shards:

```json
{
   "parquet_files":[
      {
         "dataset":"amazon_polarity",
         "config":"amazon_polarity",
         "split":"test",
         "url":"https://huggingface.co/datasets/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/test/0000.parquet",
         "filename":"0000.parquet",
         "size":117422360
      },
      {
         "dataset":"amazon_polarity",
         "config":"amazon_polarity",
         "split":"train",
         "url":"https://huggingface.co/datasets/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/train/0000.parquet",
         "filename":"0000.parquet",
         "size":259761770
      },
      {
         "dataset":"amazon_polarity",
         "config":"amazon_polarity",
         "split":"train",
         "url":"https://huggingface.co/datasets/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/train/0001.parquet",
         "filename":"0001.parquet",
         "size":258363554
      },
      {
         "dataset":"amazon_polarity",
         "config":"amazon_polarity",
         "split":"train",
         "url":"https://huggingface.co/datasets/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/train/0002.parquet",
         "filename":"0002.parquet",
         "size":255471883
      },
      {
         "dataset":"amazon_polarity",
         "config":"amazon_polarity",
         "split":"train",
         "url":"https://huggingface.co/datasets/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/train/0003.parquet",
         "filename":"0003.parquet",
         "size":254410930
      }
   ],
   "pending":[
      
   ],
   "failed":[
      
   ],
   "partial":false
}
```

To read and query the Parquet files, take a look at the [Query datasets from the dataset viewer API](parquet_process) guide.

## Partially converted datasets

The Parquet version can be partial in two cases:
- if the dataset is already in Parquet format but it contains row groups bigger than the recommended size (100-300MB uncompressed)
- if the dataset is not already in Parquet format or if it is bigger than 5GB.

In that case the Parquet files are generated up to 5GB and placed in a split directory prefixed with "partial", e.g. "partial-train" instead of "train".

You can check the row groups size directly on Hugging Face using the Parquet metadata sidebar, for example [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/data/CC-MAIN-2013-20?show_file_info=data%2FCC-MAIN-2013-20%2Ftrain-00000-of-00014.parquet):

![clic-parquet-metadata-sidebar](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/clic-parquet-metadata-sidebar.png)

![parquet-metadata-sidebar-total-byte-size](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/parquet-metadata-sidebar-total-byte-size.png)

## Parquet-native datasets

When the dataset is already in Parquet format, the data are not converted and the files in `refs/convert/parquet` are links to the original files. This rule suffers an exception to ensure the dataset viewer API to stay fast: if the [row group](https://parquet.apache.org/docs/concepts/) size of the original Parquet files is too big, new Parquet files are generated.

## Using the Hugging Face Hub API

For convenience, you can directly use the Hugging Face Hub `/api/parquet` endpoint which returns the list of Parquet URLs:

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://huggingface.co/api/datasets/ibm/duorc/parquet"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
urls = query()
```
</python>
<js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://huggingface.co/api/datasets/ibm/duorc/parquet",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const urls = await response.json();
    return urls;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js>
<curl>
```curl
curl https://huggingface.co/api/datasets/ibm/duorc/parquet \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The endpoint response is a JSON containing a list of the dataset's files URLs in the Parquet format for each split and configuration. For example, the [`ibm/duorc`](https://huggingface.co/datasets/ibm/duorc) dataset has one Parquet file for the train split of the "ParaphraseRC" configuration (see the [List splits and configurations](./splits) guide for more details about splits and configurations).

```json
{
   "ParaphraseRC":{
      "test":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/test/0.parquet"
      ],
      "train":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train/0.parquet"
      ],
      "validation":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/validation/0.parquet"
      ]
   },
   "SelfRC":{
      "test":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/SelfRC/test/0.parquet"
      ],
      "train":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/SelfRC/train/0.parquet"
      ],
      "validation":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/SelfRC/validation/0.parquet"
      ]
   }
}
```

Optionally you can specify which configuration name to return, as well as which split:

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
urls = query()
```
</python>
<js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const urls = await response.json();
    return urls;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js>
<curl>
```curl
curl https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

```json
[
  "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train/0.parquet"
]
```

Each parquet file can also be accessed using its shard index: `https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train/0.parquet` redirects to `https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/ParaphraseRC/train/0000.parquet` for example.
