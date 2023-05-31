from datasets import load_dataset
from elasticsearch import Elasticsearch
from datetime import datetime

duorc = load_dataset("LLMs/Alpaca-ShareGPT", split="train")
es = Elasticsearch("http://localhost:9200")
start_time = datetime.now()

for i, row in enumerate(duorc):
    doc = {
        "config": "LLMs--Alpaca-ShareGPT",
        "split": "train",
        "index": i,
        "row": row,
    }
            
    es.index(index="LLMs--Alpaca-ShareGPT".lower(), id=i, document=doc)
    print(f"indexed row {i}")
end_time = datetime.now()
print(f"Duration: {end_time - start_time}")