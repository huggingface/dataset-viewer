
from typing import List
import duckdb
import pandas as pd
import requests
from datetime import datetime

DATASETS_SERVER_ENDPOINT = "https://datasets-server.huggingface.co"
PARQUET_REVISION="refs/convert/parquet"

EXAMPLE_DATASET_NAME = "LLMs/Alpaca-ShareGPT"

con = duckdb.connect('datasets-server.db')

def get_parquet_urls(dataset: str) -> List[str]:
        splits = requests.get(f"{DATASETS_SERVER_ENDPOINT}/splits?dataset={dataset}", timeout=60).json().get("splits")
        split = splits[0]
        response = requests.get(f"{DATASETS_SERVER_ENDPOINT}/parquet?dataset={dataset}&config={split['config']}", timeout=60)
        if response.status_code != 200:
            raise Exception(response)
        
        response = response.json()
        parquet_files = response["parquet_files"]
        urls = [content["url"] for content in parquet_files if content["split"] == split["split"]]
        if len(urls) == 0:
             raise Exception("No parquet files found for dataset")
        return urls

def import_data():
    start_time = datetime.now()

    duckdb.execute("INSTALL 'httpfs';")
    duckdb.execute("LOAD 'httpfs';")
    duckdb.execute("INSTALL 'fts';")
    duckdb.execute("LOAD 'fts';")
    # duckdb.sql("select * from duckdb_extensions();").show()
    
    # Import data + index
    parquet_url = get_parquet_urls(EXAMPLE_DATASET_NAME)[0]
    print("parquet_url", parquet_url)
    con.sql("CREATE SEQUENCE serial START 1;")
    # We need a sequence id column for Full text search
    # I'm very rusty in SQL so it's very possible there are simpler ways.

    con.sql(f"CREATE TABLE data AS SELECT nextval('serial') AS id, * FROM '{parquet_url}';")
    con.sql("PRAGMA create_fts_index('data', 'id', '*');")

    con.sql("DESCRIBE SELECT * FROM data").show()
    end_time = datetime.now()
    print(f"Duration: {end_time - start_time}")

import_data()