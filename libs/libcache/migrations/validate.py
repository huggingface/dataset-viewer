from libcache.cache import DbSplit, connect_to_cache
from ._utils import check_documents, MONGO_CACHE_DATABASE, MONGO_URL


connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
check_documents(DbSplit, 100)
