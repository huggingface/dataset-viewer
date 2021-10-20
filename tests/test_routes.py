from datasets_preview_backend.queries.rows import get_rows
from datasets_preview_backend.routes import get_response


def test_datetime_content() -> None:
    get_response(get_rows, 0, dataset_name="allenai/c4")
