from datetime import datetime

from datasets_preview_backend.io.cache import DbRow, Status, connect_to_cache
from datasets_preview_backend.io.migrations._utils import check_documents

# connect
connect_to_cache()

# migrate
DbRow.objects().update(status=Status.VALID, since=datetime.utcnow)


# validate
def custom_validation(row: DbRow) -> None:
    if row.status != Status.VALID:
        raise ValueError(f"row status should be '{Status.VALID}', got '{row.status}'")


check_documents(DbRow, 100, custom_validation)
