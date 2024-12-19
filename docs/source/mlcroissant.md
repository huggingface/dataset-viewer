# mlcroissant

[mlcroissant](https://github.com/mlcommons/croissant/tree/main/python/mlcroissant) is a library to load datasets from Croissant metadata.

<Tip>

💡 Learn more about how to get the metadata from the dataset viewer API in the [Get Croissant metadata](croissant) guide.

</Tip>

Let's start by parsing the Croissant metadata for the [`tasksource/blog_authorship_corpus`](https://huggingface.co/datasets/tasksource/blog_authorship_corpus) dataset. Be sure to first install `mlcroissant[parquet]` and `GitPython` to be able to load Parquet files over the git+https protocol.

```py
from mlcroissant import Dataset
ds = Dataset(jsonld="https://huggingface.co/api/datasets/tasksource/blog_authorship_corpus/croissant")
```

To read from the first subset (called RecordSet in Croissant's vocabulary), use the [`records`](https://github.com/mlcommons/croissant/blob/cd64e12c733cf8bf48f2f85c951c1c67b1c94f5a/python/mlcroissant/mlcroissant/_src/datasets.py#L86) function, which returns an iterator of dicts.

```py
records = ds.records("default")
```

Finally use Pandas to compute your query on the first 1,000 rows:

```py
import itertools

import pandas as pd

df = (
    pd.DataFrame(list(itertools.islice(records, 100)))
    .groupby("default/sign")["default/text"]
    .apply(lambda x: x.str.len().mean())
    .sort_values(ascending=False)
    .head(5)
)
print(df)
default/sign
b'Leo'          6463.500000
b'Capricorn'    2374.500000
b'Aquarius'     2303.757143
b'Gemini'       1420.333333
b'Aries'         918.666667
Name: default/text, dtype: float64
```
