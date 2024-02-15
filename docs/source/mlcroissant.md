# mlcroissant

[mlcroissant](https://github.com/mlcommons/croissant/tree/main/python/mlcroissant) is a library to load datasets from Croissant metadata.

<Tip>

💡 Learn more about how to get the Parquet files in the [Get Croissant metadata](croissant) guide.

</Tip>

Let's start by parsing the Croissant metadata for the [`blog_authorship_corpus`](https://huggingface.co/datasets/blog_authorship_corpus) dataset from Datasets Server. Be sure to first install `mlcroissant[parquet]` and `GitPython` to be able to load Parquet files over the git+https protocol.

```py
from mlcroissant import Dataset
ds = Dataset(jsonld="https://datasets-server.huggingface.co/croissant?dataset=blog_authorship_corpus")
```

To read from the first subset (called RecordSet in Croissant's vocabulary), use the [`records`](https://github.com/mlcommons/croissant/blob/cd64e12c733cf8bf48f2f85c951c1c67b1c94f5a/python/mlcroissant/mlcroissant/_src/datasets.py#L86) function to read it into a dict (let's only take the first 1,000 rows).

```py
import itertools

records = list(itertools.islice(ds.records(ds.metadata.record_sets[0].uid), 1000))
```

Finally use Pandas to compute your query:

```py
import pandas as pd

df = (
    pd.DataFrame(records)
    .groupby("horoscope")["text"]
    .apply(lambda x: x.str.len().mean())
    .sort_values(ascending=False)
    .head(5)
)
print(df)
horoscope
b'Sagittarius'    1216.000000
b'Libra'           862.615581
b'Capricorn'       381.269231
b'Cancer'          272.776471
Name: text, dtype: float64
```
