# DuckDB

The [DuckDB CLI](https://duckdb.org/docs/api/cli/overview.html) (Command Line Interface) is a single, dependency-free executable. 

<!-- <Tip>

For installation details, visit the [installation page](https://duckdb.org/docs/installation).

</Tip> -->

Starting from version `v0.10.3-dev1012`, the DuckDB CLI includes native support for accessing datasets on Hugging Face via URLs. Here are some features you can leverage with this powerful tool:

- Query public, gated and private datasets
- Analyze datasets and perform SQL operations
- Process and transform datasets
- Conduct vector similarity search on embedding datasets
- Export datasets to other formats
- Implement full-text search on datasets
- And more! For a complete list of DuckDB features, visit the DuckDB documentation.

Let's start with a quick demo to query the full rows of a dataset under the `refs/convert/parquet` revision:

```bash
FROM 'hf://datasets/ibm/duorc@~parquet/**/*.parquet';
```

#TODO: Put an image of the output?

In the following sections, we will cover more complex operations you can perform with DuckDB on Hugging Face datasets.
