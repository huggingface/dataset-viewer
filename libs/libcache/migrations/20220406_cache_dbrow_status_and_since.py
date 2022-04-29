from datetime import datetime

from pymongo import MongoClient

from libcache.cache import Status
from ._utils import MONGO_CACHE_DATABASE, MONGO_URL

client = MongoClient(MONGO_URL)
db = client[MONGO_CACHE_DATABASE]


# migrate
rows_coll = db.rows
rows_coll.update_many({}, {"$set": {"status": Status.VALID.value, "since": datetime.utcnow}})
