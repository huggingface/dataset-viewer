## Datasets-server Admin UI

Deployed at (internal) https://huggingface.co/spaces/datasets-maintainers/datasets-server-admin-ui

### Setup:

```
poetry install
```

### Run:

To connect to the PROD endpoint:

```
poetry run python app.py
```

To connect to your local DEV endpoint:

```
DEV=1 HF_TOKEN=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 poetry run python app.py
```
