from datetime import datetime

from pymongo import MongoClient

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_URL
from datasets_preview_backend.io.cache import Status

client = MongoClient(MONGO_URL)
db = client[MONGO_CACHE_DATABASE]


# migrate
rows_coll = db.rows
rows_coll.update_many({}, {"$set": {"status": Status.VALID.value, "since": datetime.utcnow}})
