# PostgreSQL

[PostgreSQL](https://www.postgresql.org/docs/) is a powerful, open source object-relational database system. It is the most [popular](https://survey.stackoverflow.co/2024/technology#most-popular-technologies-database) database by application developers for a few years running. [pgai](https://github.com/timescale/pgai) is a PostgreSQL extension that allows you to easily ingest huggingface datasets into your PostgreSQL database.


## Run PostgreSQL with pgai installed

You can easily run a docker container containing PostgreSQL with pgai.

```bash
docker run -d --name pgai -p 5432:5432 \
-v pg-data:/home/postgres/pgdata/data \
-e POSTGRES_PASSWORD=password timescale/timescaledb-ha:pg17
```

Then run the following command to install pgai into the database.

```bash
docker exec -it pgai psql -c "CREATE EXTENSION ai CASCADE;"
```

You can then connect to the database using the `psql` command line tool in the container.

```bash
docker exec -it pgai psql
```

or using your favorite PostgreSQL client using the following connection string: `postgresql://postgres:password@localhost:5432/postgres
`

Alternatively, you can install pgai into an existing PostgreSQL database. For instructions on how to install pgai into an existing PostgreSQL database, follow the instructions in the [github repo](https://github.com/timescale/pgai).

## Create a table from a dataset

To load a dataset into PostgreSQL, you can use the `ai.load_dataset` function. This function will create a PostgreSQL table, and load the dataset from the Hugging Face Hub
in a streaming fashion.

```sql
select ai.load_dataset('rajpurkar/squad', table_name => 'squad');
```

You can now query the table using standard SQL.

```sql
select * from squad limit 10;
```

Full documentation for the `ai.load_dataset` function can be found [here](https://github.com/timescale/pgai/blob/main/docs/load_dataset_from_huggingface.md).

## Import only a subset of the dataset

You can also import a subset of the dataset by specifying the `max_batches` parameter.
This is useful if the dataset is large and you want to experiment with a smaller subset.

```sql
SELECT ai.load_dataset('rajpurkar/squad', table_name => 'squad', batch_size => 100, max_batches => 1);
```

## Load a dataset into an existing table

You can also load a dataset into an existing table.
This is useful if you want more control over the data schema or want to predefine indexes and constraints on the data.

```sql
select ai.load_dataset('rajpurkar/squad', table_name => 'squad', if_table_exists => 'append');
```
