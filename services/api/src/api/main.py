from api.api import start_api
from api.metrics import start_metrics

if __name__ == "__main__":
    start_metrics()
    start_api()
