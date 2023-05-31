import duckdb
import pandas as pd

DATASETS_SERVER_ENDPOINT = "https://datasets-server.huggingface.co"
PARQUET_REVISION="refs/convert/parquet"

EXAMPLE_DATASET_NAME = "LLMs/Alpaca-ShareGPT"

con = duckdb.connect('datasets-server.db')

def run_command(query: str) -> pd.DataFrame:
    try:
        result = con.execute("SELECT fts_main_data.match_bm25(id, ?) AS score, id, instruction, input, output   FROM data   WHERE score IS NOT NULL   ORDER BY score DESC;", [query])
        print("Ok")
    except Exception as error:
        print(f"Error: {str(error)}")
        return pd.DataFrame({"Error": [f"❌ {str(error)}"]})
    print(result)
    return result.df()

result = run_command("Jonny Walker")
print(result)