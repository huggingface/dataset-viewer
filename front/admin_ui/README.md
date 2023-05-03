---
title: Datasets Server Admin UI
emoji: 📊
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 3.28.1
python_version: 3.9.15
app_file: app.py
pinned: false
---

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
or to enable auto reloading:
```
DEV=1 HF_TOKEN=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 poetry run gradio app.py
```
