install:
	poetry install

run:
	poetry run python datasets_preview_backend/main.py

watch:
	poetry run watchmedo auto-restart -d datasets_preview_backend -p "*.py" -R python datasets_preview_backend/main.py
