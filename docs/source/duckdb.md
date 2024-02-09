# DuckDB

[DuckDB](https://duckdb.org/docs/) is a database that supports reading and querying Parquet files really fast. Begin by creating a connection to DuckDB, and then install and load the [`httpfs`](https://duckdb.org/docs/extensions/httpfs.html) extension to read and write remote files:

<inferencesnippet>
<python>
```py
import duckdb

url = "https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0000.parquet"

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

const url = "https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/train/0000.parquet"
```
</js>
</inferencesnippet>

Now you can write and execute your SQL query on the Parquet file:

<inferencesnippet>
<python>
```py
con.sql(f"SELECT horoscope, count(*), AVG(LENGTH(text)) AS avg_blog_length FROM '{url}' GROUP BY horoscope ORDER BY avg_blog_length DESC LIMIT(5)")
┌───────────┬──────────────┬────────────────────┐
│ horoscope │ count_star() │  avg_blog_length   │
│  varchar  │    int64     │       double       │
├───────────┼──────────────┼────────────────────┤
│ Aquarius  │        34062 │  1129.218836239798 │
│ Cancer    │        41509 │  1098.366812016671 │
│ Capricorn │        33961 │ 1073.2002002296751 │
│ Libra     │        40302 │ 1072.0718326633914 │
│ Leo       │        40587 │ 1064.0536871412028 │
└───────────┴──────────────┴────────────────────┘
```
</python>
<js>
```js
con.all(`SELECT horoscope, count(*), AVG(LENGTH(text)) AS avg_blog_length FROM '${url}' GROUP BY horoscope ORDER BY avg_blog_length DESC LIMIT(5)`, function(err, res) {
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
con.sql(f"SELECT horoscope, count(*), AVG(LENGTH(text)) AS avg_blog_length FROM read_parquet({urls[:2]}) GROUP BY horoscope ORDER BY avg_blog_length DESC LIMIT(5)")
┌─────────────┬──────────────┬────────────────────┐
│  horoscope  │ count_star() │  avg_blog_length   │
│   varchar   │    int64     │       double       │
├─────────────┼──────────────┼────────────────────┤
│ Aquarius    │        49568 │ 1125.8306770497095 │
│ Cancer      │        63512 │   1097.95608703867 │
│ Libra       │        60304 │ 1060.6110539931017 │
│ Capricorn   │        49402 │ 1059.5552609206104 │
│ Sagittarius │        50431 │ 1057.4589835616982 │
└─────────────┴──────────────┴────────────────────┘
```
</python>
<js>
```js
con.all(`SELECT horoscope, count(*), AVG(LENGTH(text)) AS avg_blog_length FROM read_parquet(${JSON.stringify(urls)}) GROUP BY horoscope ORDER BY avg_blog_length DESC LIMIT(5)`, function(err, res) {
  if (err) {
    throw err;
  }
  console.log(res)
});
```
</js>
</inferencesnippet>

[DuckDB-Wasm](https://duckdb.org/docs/api/wasm), a package powered by [WebAssembly](https://webassembly.org/), is also available for running DuckDB in any browser. This could be useful, for instance, if you want to create a web app to query Parquet files from the browser!
