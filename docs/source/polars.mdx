# Polars 

[Polars](https://pola-rs.github.io/polars-book/user-guide/) is a fast DataFrame library written in Rust with Arrow as its foundation.

<Tip>

💡 Learn more about how to get the dataset URLs in the [List Parquet files](parquet) guide.

</Tip>

Let's start by grabbing the URLs to the `train` split of the [`blog_authorship_corpus`](https://huggingface.co/datasets/blog_authorship_corpus) dataset from Datasets Server:

```py
r = requests.get("https://datasets-server.huggingface.co/parquet?dataset=blog_authorship_corpus")
j = r.json()
urls = [f['url'] for f in j['parquet_files'] if f['split'] == 'train']
urls
['https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0000.parquet',
 'https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0001.parquet']
```

To read from a single Parquet file, use the [`read_parquet`](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.read_parquet.html) function to read it into a DataFrame and then execute your query:

```py
import polars as pl

df = (
    pl.read_parquet("https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0000.parquet")
    .groupby("horoscope")
    .agg(
        [
            pl.count(),
            pl.col("text").str.n_chars().mean().alias("avg_blog_length")
        ]
    )
    .sort("avg_blog_length", descending=True)
    .limit(5)
)
print(df)
shape: (5, 3)
┌───────────┬───────┬─────────────────┐
│ horoscope ┆ count ┆ avg_blog_length │
│ ---       ┆ ---   ┆ ---             │
│ str       ┆ u32   ┆ f64             │
╞═══════════╪═══════╪═════════════════╡
│ Aquarius  ┆ 34062 ┆ 1129.218836     │
│ Cancer    ┆ 41509 ┆ 1098.366812     │
│ Capricorn ┆ 33961 ┆ 1073.2002       │
│ Libra     ┆ 40302 ┆ 1072.071833     │
│ Leo       ┆ 40587 ┆ 1064.053687     │
└───────────┴───────┴─────────────────┘
```

To read multiple Parquet files - for example, if the dataset is sharded - you'll need to use the [`concat`](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.concat.html) function to concatenate the files into a single DataFrame: 

```py
import polars as pl
df = (
    pl.concat([pl.read_parquet(url) for url in urls])
    .groupby("horoscope")
    .agg(
        [
            pl.count(),
            pl.col("text").str.n_chars().mean().alias("avg_blog_length")
        ]
    )
    .sort("avg_blog_length", descending=True)
    .limit(5)
)
print(df)
shape: (5, 3)
┌─────────────┬───────┬─────────────────┐
│ horoscope   ┆ count ┆ avg_blog_length │
│ ---         ┆ ---   ┆ ---             │
│ str         ┆ u32   ┆ f64             │
╞═════════════╪═══════╪═════════════════╡
│ Aquarius    ┆ 49568 ┆ 1125.830677     │
│ Cancer      ┆ 63512 ┆ 1097.956087     │
│ Libra       ┆ 60304 ┆ 1060.611054     │
│ Capricorn   ┆ 49402 ┆ 1059.555261     │
│ Sagittarius ┆ 50431 ┆ 1057.458984     │
└─────────────┴───────┴─────────────────┘
```

## Lazy API

Polars offers a [lazy API](https://pola-rs.github.io/polars-book/user-guide/lazy/using/) that is more performant and memory-efficient for large Parquet files. The LazyFrame API keeps track of what you want to do, and it'll only execute the entire query when you're ready. This way, the lazy API doesn't load everything into RAM beforehand, and it allows you to work with datasets larger than your available RAM.

To lazily read a Parquet file, use the [`scan_parquet`](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.scan_parquet.html) function instead. Then, execute the entire query with the [`collect`](https://pola-rs.github.io/polars/py-polars/html/reference/lazyframe/api/polars.LazyFrame.collect.html) function:

```py
import polars as pl

q = (
    pl.scan_parquet("https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0000.parquet")
    .groupby("horoscope")
    .agg(
        [
            pl.count(),
            pl.col("text").str.n_chars().mean().alias("avg_blog_length")
        ]
    )
    .sort("avg_blog_length", descending=True)
    .limit(5)
)
df = q.collect()
```