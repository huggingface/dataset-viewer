# Query public, gated and private datasets

Querying datasets is a fundamental step in data analysis. Here, we'll guide you through querying datasets using various methods.

You can query Hugging Face [autoconverted parquet files](https://huggingface.co/docs/datasets-server/en/parquet#conversion-to-parquet) in the `refs/converts/parquet` branch by using the following syntax:


```plaintext
hf://datasets/{my-username}/{my-dataset}@~parquet/{path_to_parquet_file} 
```

There are [different ways](https://duckdb.org/docs/data/parquet/overview.html) to select your data.

Using `FROM` syntax:
```bash
FROM 'hf://datasets/ibm/duorc@~parquet/ParaphraseRC/train/0000.parquet';
```

Using `SELECT` `FROM` sytax:

```bash
SELECT question, answers FROM 'hf://datasets/ibm/duorc@~parquet/ParaphraseRC/train/0000.parquet' LIMIT 10;
```

Count all parquet files matching a glob pattern:

```bash
SELECT COUNT(*) FROM 'hf://datasets/ibm/duorc@~parquet/**/*.parquet';
```

Select using [read_parquet](https://duckdb.org/docs/guides/file_formats/query_parquet.html) function:

```bash
SELECT * FROM read_parquet('hf://datasets/ibm/duorc@~parquet/ParaphraseRC/**/*.parquet') LIMIT 10;
```

Read all files that match a glob pattern and include a filename column specifying which file each row came from:

```bash
SELECT * FROM read_parquet('hf://datasets/ibm/duorc@~parquet/ParaphraseRC/**/*.parquet', filename = true) LIMIT 10;
```

Using `parquet_scan` function:

```bash
SELECT * FROM parquet_scan('hf://datasets/ibm/duorc@~parquet/ParaphraseRC/**/*.parquet') LIMIT 10;
```

## Get information of parquet files

The [parquet_metadata]((https://duckdb.org/docs/data/parquet/metadata.html)) function can be used to query the metadata contained within a Parquet file.

```bash
SELECT * FROM parquet_metadata('hf://datasets/ibm/duorc@~parquet/ParaphraseRC/train/0000.parquet');
```

Fetch the column names and column types:

```bash
DESCRIBE SELECT * FROM 'hf://datasets/ibm/duorc@~parquet/ParaphraseRC/train/0000.parquet';
```

Fetch the internal schema:

```bash
SELECT * FROM parquet_schema('hf://datasets/ibm/duorc@~parquet/ParaphraseRC/train/0000.parquet');
```

## Get statistics of the parquet files

The `SUMMARIZE` command can be used to get various aggregates over a query (min, max, approx_unique, avg, std, q25, q50, q75, count). It returns these along with the column name, column type, and the percentage of NULL values.

```bash
SUMMARIZE SELECT * FROM 'hf://datasets/ibm/duorc@~parquet/ParaphraseRC/train/0000.parquet';
```
