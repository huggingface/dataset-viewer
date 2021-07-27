install:
	poetry install

run:
	poetry run python src/datasets_preview_backend/main.py

watch:
	poetry run watchmedo auto-restart -d src/datasets_preview_backend -p "*.py" -R python src/datasets_preview_backend/main.py
