from api.app import start
from api.metrics import start_metrics

if __name__ == "__main__":
    start_metrics()
    start()
