from datasets_preview_backend.io.cache import DbSplit, connect_to_cache
from datasets_preview_backend.io.migrations._utils import check_documents

connect_to_cache()
check_documents(DbSplit, 100)
