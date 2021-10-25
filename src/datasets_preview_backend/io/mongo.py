from mongoengine import connect

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_URL


def connect_cache() -> None:
    connect(MONGO_CACHE_DATABASE, host=MONGO_URL)
