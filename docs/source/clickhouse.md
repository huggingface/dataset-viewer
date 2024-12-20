# ClickHouse

[ClickHouse](https://clickhouse.com/docs/en/intro) is a fast and efficient column-oriented database for analytical workloads, making it easy to analyze Hub-hosted datasets with SQL. To get started quickly, use [`clickhouse-local`](https://clickhouse.com/docs/en/operations/utilities/clickhouse-local) to run SQL queries from the command line and avoid the need to fully install ClickHouse.

<Tip>

Check this [blog](https://clickhouse.com/blog/query-analyze-hugging-face-datasets-with-clickhouse) for more details about how to analyze datasets on the Hub with ClickHouse.

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
['https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet']
```

## Aggregate functions

Now you can begin to analyze the dataset. Use the `-q` argument to specify the query to execute, and the [`url`](https://clickhouse.com/docs/en/sql-reference/table-functions/url) function to create a table from the data in the Parquet file.

You should set `enable_url_encoding` to 0 to ensure the escape characters in the URL are preserved as intended, and `max_https_get_redirects` to 1 to redirect to the path of the Parquet file.

Let's start by identifying the most popular artists:

```bash
./clickhouse local -q "
    SELECT count() AS c, artists 
    FROM url('https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet') 
    GROUP BY artists 
    ORDER BY c 
    DESC LIMIT 5
    SETTINGS enable_url_encoding=0, max_http_get_redirects=1"

┌───c─┬─artists─────────┐
│ 279 │ The Beatles 	│
│ 271 │ George Jones	│
│ 236 │ Stevie Wonder   │
│ 224 │ Linkin Park 	│
│ 222 │ Ella Fitzgerald │
└─────┴─────────────────┘
```

ClickHouse also provides functions for visualizing your queries. For example, you can use the [`bar`](https://clickhouse.com/docs/en/sql-reference/functions/other-functions#bar) function to create a bar chart of the danceability of songs:

```bash
./clickhouse local -q "
    SELECT
        round(danceability, 1) AS danceability,
        bar(count(), 0, max(count()) OVER ()) AS dist
    FROM url('https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet')
    GROUP BY danceability
    ORDER BY danceability ASC
    SETTINGS enable_url_encoding=0, max_http_get_redirects=1"

┌─danceability─┬─dist─────────────────────────────────────────────────────────────────────────────────┐
│            0 │ ▍                                                                            	      │
│      	   0.1 │ ████▎                                                                        	      │
│      	   0.2 │ █████████████▍                                                               	      │
│      	   0.3 │ ████████████████████████                                                     	      │
│      	   0.4 │ ████████████████████████████████████████████▋                                	      │
│      	   0.5 │ ████████████████████████████████████████████████████████████████████▊        	      │
│      	   0.6 │ ████████████████████████████████████████████████████████████████████████████████     │
│      	   0.7 │ ██████████████████████████████████████████████████████████████████████       	      │
│      	   0.8 │ ██████████████████████████████████████████                                   	      │
│      	   0.9 │ ██████████▋                                                                  	      │
│            1 │ ▌                                                                            	      │
└──────────────┴──────────────────────────────────────────────────────────────────────────────────────┘
```

To get a deeper understanding about a dataset, ClickHouse provides statistical analysis functions for determining how your data is correlated, calculating statistical hypothesis tests, and more. Take a look at ClickHouse's [List of Aggregate Functions](https://clickhouse.com/docs/en/sql-reference/aggregate-functions/reference) for a complete list of available aggregate functions.

## User-defined function (UDFs)

A user-defined function (UDF) allows you to reuse custom logic. Many Hub datasets are often sharded into more than one Parquet file, so it can be easier and more efficient to create a UDF to list and query all the Parquet files of a given dataset from just the dataset name.

For this example, you'll need to run `clickhouse-local` in console mode so the UDF persists between queries:

```bash
./clickhouse local
```

Remember to set `enable_url_encoding` to 0 and `max_https_get_redirects` to 1 to redirect to the path of the Parquet files:

```bash
SET max_http_get_redirects = 1, enable_url_encoding = 0
```

Let's create a function to return a list of Parquet files from the [`tasksource/blog_authorship_corpus`](https://huggingface.co/datasets/tasksource/blog_authorship_corpus):

```bash
CREATE OR REPLACE FUNCTION hugging_paths AS dataset -> (
    SELECT arrayMap(x -> (x.1), JSONExtract(json, 'parquet_files', 'Array(Tuple(url String))'))
    FROM url('https://datasets-server.huggingface.co/parquet?dataset=' || dataset, 'JSONAsString')
);

SELECT hugging_paths('tasksource/blog_authorship_corpus') AS paths

['https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet','https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet']
```

You can make this even easier by creating another function that calls `hugging_paths` and outputs all the files based on the dataset name:

```bash
CREATE OR REPLACE FUNCTION hf AS dataset -> (
    WITH hugging_paths(dataset) as urls
    SELECT multiIf(length(urls) = 0, '', length(urls) = 1, urls[1], 'https://huggingface.co/datasets/{' || arrayStringConcat(arrayMap(x -> replaceRegexpOne(replaceOne(x, 'https://huggingface.co/datasets/', ''), '\\.parquet$', ''), urls), ',') || '}.parquet')
);

SELECT hf('tasksource/blog_authorship_corpus') AS pattern

https://huggingface.co/datasets/{tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0000,tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0001}.parquet 
```

Now use the `hf` function to query any dataset by passing the dataset name:

```bash
SELECT sign, count(*), AVG(LENGTH(text)) AS avg_blog_length 
FROM url(hf('tasksource/blog_authorship_corpus'))
GROUP BY sign 
ORDER BY avg_blog_length 
DESC LIMIT(5) 

┌───────────┬────────┬────────────────────┐
│  sign     │ count  │ avg_blog_length    │
├───────────┼────────┼────────────────────┤
│ Aquarius  │ 49687  │ 1193.9523819107615 │
│ Leo       │ 53811  │ 1186.0665291483153 │
│ Cancer    │ 65048  │ 1160.8010392325666 │
│ Gemini    │ 51985  │ 1158.4132922958545 │
│ Vurgi     │ 60399  │ 1142.9977648636566 │
└───────────┴────────┴────────────────────┘
```
