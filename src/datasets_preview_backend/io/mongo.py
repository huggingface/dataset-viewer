from pymongo import MongoClient

from datasets_preview_backend.config import MONGO_URL


def get_client() -> MongoClient:
    return MongoClient(MONGO_URL)
