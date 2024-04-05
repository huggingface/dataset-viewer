# mlcroissant

[mlcroissant](https://github.com/mlcommons/croissant/tree/main/python/mlcroissant) is a library to load datasets from Croissant metadata.

<Tip>

💡 Learn more about how to get the metadata from the dataset viewer API in the [Get Croissant metadata](croissant) guide.

</Tip>

Let's start by parsing the Croissant metadata for the [`blog_authorship_corpus`](https://huggingface.co/datasets/blog_authorship_corpus) dataset. Be sure to first install `mlcroissant[parquet]` and `GitPython` to be able to load Parquet files over the git+https protocol.

```py
from mlcroissant import Dataset
ds = Dataset(jsonld="https://huggingface.co/api/datasets/blog_authorship_corpus/croissant")
```

To read from the first subset (called RecordSet in Croissant's vocabulary), use the [`records`](https://github.com/mlcommons/croissant/blob/cd64e12c733cf8bf48f2f85c951c1c67b1c94f5a/python/mlcroissant/mlcroissant/_src/datasets.py#L86) function, which returns an iterator of dicts.

```py
records = ds.records(ds.metadata.record_sets[0].uid)
```

Finally use Pandas to compute your query on the first 1,000 rows:

```py
import itertools

import pandas as pd

df = (
    pd.DataFrame(list(itertools.islice(records, 1000)))
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
