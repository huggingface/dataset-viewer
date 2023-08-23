# ClickHouse

[ClickHouse](https://clickhouse.com/docs/en/intro) is a fast and efficient column-oriented database for analytical workloads, making it easy to analyze Hub hosted datasets with SQL. To get started quickly, use [`clickhouse-local`](https://clickhouse.com/docs/en/operations/utilities/clickhouse-local) to run SQL queries from the command line and avoid the need to fully install ClickHouse.

<Tip>

Learn more about the Hugging Face and ClickHouse integration in this blog post for more details about how to analyze datasets on the Hub with ClickHouse.

</Tip>

To start, download and install `clickhouse-local`:

```bash
curl https://clickhouse.com/ | sh
```

For this example, you'll analyze the [maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) which contains information about Spotify tracks. Datasets on the Hub are stored as Parquet files and you can access it with the [`/parquet`](parquet) endpoint:

```py
import requests

r = requests.get("https://datasets-server.huggingface.co/parquet?dataset=maharshipandya/spotify-tracks-dataset")
j = r.json()
url = [f['url'] for f in j['parquet_files']]
url
['https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/maharshipandya--spotify-tracks-dataset/csv-train.parquet']
```

## Aggregate functions

Now you can begin to analyze the dataset. Use the `-q` argument to specify the query to execute, and the [`url`](https://clickhouse.com/docs/en/sql-reference/table-functions/url) function to create a table from the data in the Parquet file.

Let's start by identifying the most popular artists:

```bash
./clickhouse local -q "
    SELECT count() AS c, artists 
    FROM url('https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/maharshipandya--spotify-tracks-dataset/csv-train.parquet') 
    GROUP BY artists 
    ORDER BY c 
    DESC LIMIT 20"
```

ClickHouse also provides functions for visualizing your queries. For example, you can use the [`sparkBar`](https://clickhouse.com/docs/en/sql-reference/aggregate-functions/reference/sparkbar) function to create a histogram of track length for each music genre:

```bash
./clickhouse local -q "
    SELECT
        track_genre,
        sparkbar(40)(CAST(duration_ms, 'UInt32'), c) AS distribution
    FROM
    (
        SELECT
            track_genre,
            count() AS c,
            duration_ms
        FROM url('https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/maharshipandya--spotify-tracks-dataset/csv-train.parquet')
        GROUP BY
            track_genre,
            round(duration_ms, -4) AS duration_ms
        ORDER BY duration_ms ASC
    ) WHERE (duration_ms >= 60000) AND (duration_ms <= 600000)
    GROUP BY track_genre
    ORDER BY track_genre ASC
    DESC LIMIT 10"
```

To get a deeper understanding about a dataset, ClickHouse provides statistical analysis functions for determining how your data is correlated, calculating statistical hypothesis tests, and more. Take a look at ClickHouse's [List of Aggregate Functions](https://clickhouse.com/docs/en/sql-reference/aggregate-functions/reference) for a complete list of available aggregate functions.

## User-defined function (UDFs)

A user-defined function (UDF) allows you to reuse custom logic. Many Hub datasets are often sharded into more than one Parquet file, so it can be easier and more efficient to create a UDF to list and query all the Parquet files of a given dataset from just the dataset name.

For example, let's create a function to return a list of Parquet files for the [blog_authorship_corpus](https://huggingface.co/datasets/blog_authorship_corpus):

```bash
./clickhouse local -q "
    CREATE OR REPLACE FUNCTION hf AS dataset -> (
        SELECT arrayMap(x -> (x.1), JSONExtract(json, 'parquet_files', 'Array(Tuple(url String))'))
        FROM url('https://datasets-server.huggingface.co/parquet?dataset=' || dataset, 'JSONAsString')
    )

    SELECT hf('blog_authorship_corpus') AS paths"

['https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0000.parquet','https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0001.parquet','https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/validation/0000.parquet']
```

Instead of passing all those paths above to the `url` function, you can just pass the dataset name to the `hf` function:

```bash
./clickhouse local -q "
    SELECT count() AS c,
        horoscope
    FROM url(hf('blog_authorship_corpus'))
    GROUP BY horoscope
    ORDER BY c DESC
    LIMIT 5"
```