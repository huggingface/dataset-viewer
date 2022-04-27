import os

from dotenv import load_dotenv
from libutils.utils import get_str_value

from libqueue.constants import DEFAULT_MONGO_QUEUE_DATABASE, DEFAULT_MONGO_URL

# Load environment variables defined in .env, if any
load_dotenv()

MONGO_QUEUE_DATABASE = get_str_value(d=os.environ, key="MONGO_QUEUE_DATABASE", default=DEFAULT_MONGO_QUEUE_DATABASE)
MONGO_URL = get_str_value(d=os.environ, key="MONGO_URL", default=DEFAULT_MONGO_URL)
