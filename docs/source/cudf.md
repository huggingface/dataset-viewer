# cuDF

[cuDF](https://docs.rapids.ai/api/cudf/stable/) is a Python GPU DataFrame library.

To read from a single Parquet file, use the [`read_parquet`](https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.read_parquet/) function to read it into a DataFrame:

```py
import cudf

df = (
    cudf.read_parquet("https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0000.parquet")
    .groupby('horoscope')['text']
    .apply(lambda x: x.str.len().mean())
    .sort_values(ascending=False)
    .head(5)
)
```

To read multiple Parquet files - for example, if the dataset is sharded - you'll need to use [`dask-cudf`](https://docs.rapids.ai/api/dask-cudf/stable/):

```py
import dask
import dask.dataframe as dd

dask.config.set({"dataframe.backend": "cudf"})

df = (
    dd.read_parquet("https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/*.parquet")
)
```