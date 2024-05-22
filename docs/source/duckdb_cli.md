# DuckDB CLI

The [DuckDB CLI](https://duckdb.org/docs/api/cli/overview.html) (Command Line Interface) is a single, dependency-free executable. 

<Tip>

For installation details, visit the [installation page](https://duckdb.org/docs/installation).

</Tip>

Starting from version `v0.10.3-dev1012`, the DuckDB CLI includes native support for accessing datasets on Hugging Face via URLs. Here are some features you can leverage with this powerful tool:

- Query public, gated and private datasets
- Analyze datasets and perform SQL operations
- Process and transform datasets
- Conduct vector similarity search on embedding datasets
- Export datasets to other formats
- Implement full-text search on datasets
- And more! For a complete list of DuckDB features, visit the DuckDB documentation.

To start the CLI, execute the following command in the installation folder:

```bash
./duckdb
```

## Forming the Hugging Face URL

To access Hugging Face datasets, use the following URL format:

```plaintext
hf://datasets/{my-username}/{my-dataset}/{path_to_parquet_file} 
```

Where:
- **my-username**  The user or organization of the dataset, e.g. `ibm`
- **my-dataset** Is the dataset name, e.g: `duorc`
- **path_to_parquet_file** Is the parquet file path, it supports glob patterns, e.g `**/*.parquet` to query all parquet files


<Tip>

You can query auto-converted Parquet files using the @~parquet branch, which corresponds to the refs/convert/parquet revision. For more details, refer to the documentation at https://huggingface.co/docs/datasets-server/en/parquet#conversion-to-parquet.

</Tip>

Let's start with a quick demo to query all the rows of a dataset:

```sql
FROM 'hf://datasets/ibm/duorc/ParaphraseRC/*.parquet' LIMIT 3;
```

Or using traditional SQL syntax:

```sql
SELECT * FROM 'hf://datasets/ibm/duorc/ParaphraseRC/*.parquet' LIMIT 3;
```
In the following sections, we will cover more complex operations you can perform with DuckDB on Hugging Face datasets.
