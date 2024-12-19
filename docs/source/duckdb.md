# DuckDB

[DuckDB](https://duckdb.org/docs/) is a database that supports reading and querying Parquet files really fast. Begin by creating a connection to DuckDB, and then install and load the [`httpfs`](https://duckdb.org/docs/extensions/httpfs.html) extension to read and write remote files:

<inferencesnippet>
<python>
```py
import duckdb

url = "https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

con = duckdb.connect()
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")
```
</python>
<js>
```js
var duckdb = require('duckdb');
var db = new duckdb.Database(':memory:');
var con = db.connect();
con.exec('INSTALL httpfs');
con.exec('LOAD httpfs');

const url = "https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
```
</js>
</inferencesnippet>

Now you can write and execute your SQL query on the Parquet file:

<inferencesnippet>
<python>
```py
con.sql(f"SELECT sign, count(*), AVG(LENGTH(text)) AS avg_blog_length FROM '{url}' GROUP BY sign ORDER BY avg_blog_length DESC LIMIT(5)")
┌───────────┬──────────────┬────────────────────┐
│   sign    │ count_star() │  avg_blog_length   │
│  varchar  │    int64     │       double       │
├───────────┼──────────────┼────────────────────┤
│ Cancer    │        38956 │ 1206.5212034089743 │
│ Leo       │        35487 │ 1180.0673767858652 │
│ Aquarius  │        32723 │ 1152.1136815084192 │
│ Virgo     │        36189 │ 1117.1982094006466 │
│ Capricorn │        31825 │  1102.397360565593 │
└───────────┴──────────────┴────────────────────┘
```
</python>
<js>
```js
con.all(`SELECT sign, count(*), AVG(LENGTH(text)) AS avg_blog_length FROM '${url}' GROUP BY sign ORDER BY avg_blog_length DESC LIMIT(5)`, function(err, res) {
  if (err) {
    throw err;
  }
  console.log(res)
});
```
</js>
</inferencesnippet>

To query multiple files - for example, if the dataset is sharded:

<inferencesnippet>
<python>
```py
urls = ["https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", "https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet"]

con.sql(f"SELECT sign, count(*), AVG(LENGTH(text)) AS avg_blog_length FROM read_parquet({urls}) GROUP BY sign ORDER BY avg_blog_length DESC LIMIT(5)")
┌──────────┬──────────────┬────────────────────┐
│   sign   │ count_star() │  avg_blog_length   │
│ varchar  │    int64     │       double       │
├──────────┼──────────────┼────────────────────┤
│ Aquarius │        49687 │  1191.417211745527 │
│ Leo      │        53811 │ 1183.8782219248853 │
│ Cancer   │        65048 │ 1158.9691612347804 │
│ Gemini   │        51985 │ 1156.0693084543618 │
│ Virgo    │        60399 │ 1140.9584430205798 │
└──────────┴──────────────┴────────────────────┘
```
</python>
<js>
```js
const urls = ["https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", "https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet"];

con.all(`SELECT sign, count(*), AVG(LENGTH(text)) AS avg_blog_length FROM read_parquet(${JSON.stringify(urls)}) GROUP BY sign ORDER BY avg_blog_length DESC LIMIT(5)`, function(err, res) {
  if (err) {
    throw err;
  }
  console.log(res)
});
```
</js>
</inferencesnippet>

[DuckDB-Wasm](https://duckdb.org/docs/api/wasm), a package powered by [WebAssembly](https://webassembly.org/), is also available for running DuckDB in any browser. This could be useful, for instance, if you want to create a web app to query Parquet files from the browser!
