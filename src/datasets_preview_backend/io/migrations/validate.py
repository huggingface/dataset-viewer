from datasets_preview_backend.io.migrations._utils import check_documents
from datasets_preview_backend.io.cache import DbSplit, connect_to_cache

connect_to_cache()
check_documents(DbSplit, 100)
