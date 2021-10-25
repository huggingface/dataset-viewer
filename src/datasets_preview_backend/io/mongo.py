from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.mongo_client import MongoClient

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_URL


def get_client() -> MongoClient:
    return MongoClient(MONGO_URL)


def get_database() -> Database:
    client = get_client()
    return client[MONGO_CACHE_DATABASE]


def get_datasets_collection() -> Collection:
    database = get_database()
    return database["datasets"]
