from pymongo import MongoClient

from ._utils import MONGO_CACHE_DATABASE, MONGO_URL

client = MongoClient(MONGO_URL)
db = client[MONGO_CACHE_DATABASE]

# migrate
db.datasets.update_many({"status": "stalled"}, {"$set": {"status": "stale"}})
db.splits.update_many({"status": "stalled"}, {"$set": {"status": "stale"}})
