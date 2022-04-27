from libcache.cache import DbSplit, connect_to_cache
from libcache.migrations._utils import check_documents

connect_to_cache()
check_documents(DbSplit, 100)
