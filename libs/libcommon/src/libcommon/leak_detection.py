import duckdb
from datasets import load_dataset

def to_duckdb_table(dataset_split):
    df = dataset_split.to_pandas()
    df['text'] = df.get('tokens', df.get('text', df.astype(str))).astype(str)
    if 'ner_tags' in df:
        df['text_label'] = df['text'] + ' ' + df['ner_tags'].astype(str)
    else:
        df['text_label'] = df['text']
    return df[['text', 'text_label']]

def detect_leaks(dataset_name, subset=None):
    print(f"Loading: {dataset_name} {f'({subset})' if subset else ''}")
    ds = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
    con = duckdb.connect()

    for split in ['train', 'validation', 'test']:
        if split in ds:
            df = to_duckdb_table(ds[split])
            con.register(split, df)

    results = {}

    queries = {
        "train_test_leaks": "SELECT COUNT(*) FROM train INNER JOIN test ON train.text = test.text",
        "validation_test_leaks": "SELECT COUNT(*) FROM validation INNER JOIN test ON validation.text = test.text",
        "train_dup": "SELECT COUNT(*) FROM (SELECT text FROM train GROUP BY text HAVING COUNT(*) > 1)",
        "validation_dup": "SELECT COUNT(*) FROM (SELECT text FROM validation GROUP BY text HAVING COUNT(*) > 1)",
        "test_dup": "SELECT COUNT(*) FROM (SELECT text FROM test GROUP BY text HAVING COUNT(*) > 1)"
    }

    for name, sql in queries.items():
        try:
            results[name] = con.execute(sql).fetchone()[0]
        except Exception as e:
            results[name] = f"error: {e}"

    return results

if __name__ == "__main__":
    stats = detect_leaks("conll2003")
    for k, v in stats.items():
        print(f"{k}: {v}")
