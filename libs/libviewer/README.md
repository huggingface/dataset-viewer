## Install

```bash
pip install maturin
maturin develop -r
```

## Index Dataset

```bash
dv --use-cache nvidia/OpenCodeReasoning index
```

This uses `huggingface_hub` to download and cache the dataset files.
Then creates a metadata file for each parque file in the dataset with
offset index included.

Remove `--use-cache` to directly download the files from the hub.

## Execute a limit/offset query

```bash
dv --use-cache nvidia/OpenCodeReasoning query --limit 10 --offset 0
```

This will query the dataset using the local metadata index files.
The scanner only reads the necessary parquet pages to minimize the
network traffic.

Remove `--use-cache` to directly query data from the hub.