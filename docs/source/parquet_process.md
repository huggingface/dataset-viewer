# Overview

The dataset viewer automatically converts and publishes public datasets less than 5GB on the Hub as Parquet files. If the dataset is already in Parquet format, it will be published as is. [Parquet](https://parquet.apache.org/docs/) files are column-based and they shine when you're working with big data.

For private datasets, the feature is provided if the repository is owned by a [PRO user](https://huggingface.co/pricing) or an [Enterprise Hub organization](https://huggingface.co/enterprise).

There are several different libraries you can use to work with the published Parquet files:

- [ClickHouse](https://clickhouse.com/docs/en/intro), a column-oriented database management system for online analytical processing
- [cuDF](https://docs.rapids.ai/api/cudf/stable/), a Python GPU DataFrame library
- [DuckDB](https://duckdb.org/docs/), a high-performance SQL database for analytical queries
- [Pandas](https://pandas.pydata.org/docs/index.html), a data analysis tool for working with data structures
- [Polars](https://pola-rs.github.io/polars-book/user-guide/), a Rust based DataFrame library
- [mlcroissant](https://github.com/mlcommons/croissant/tree/main/python/mlcroissant), a library for loading datasets from Croissant metadata
