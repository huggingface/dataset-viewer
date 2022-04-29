To warm the cache, ie. add all the missing Hugging Face datasets to the queue:

```bash
make warm
```

To empty the databases:

```bash
make clean
```

or individually:

```bash
make clean-cache
make clean-queues         # delete all the jobs
```

See also:

```bash
make cancel-started-jobs
make cancel-waiting-jobs
```

---

datasets-preview-backend is installed on a virtual machine (ec2-54-158-211-3.compute-1.amazonaws.com).

## Machine

```bash
ssh hf@ec2-54-158-211-3.compute-1.amazonaws.com

/: 200 GB

ipv4: 172.30.4.71
ipv4 (public): 54.158.211.3
domain name: datasets-preview.huggingface.tech
```

Grafana:

- https://grafana.huggingface.co/d/gBtAotjMk/use-method?orgId=2&var-DS_PROMETHEUS=HF%20Prometheus&var-node=data-preview
- https://grafana.huggingface.co/d/rYdddlPWk/node-exporter-full?orgId=2&refresh=1m&var-DS_PROMETHEUS=HF%20Prometheus&var-job=node_exporter_metrics&var-node=data-preview&var-diskdevices=%5Ba-z%5D%2B%7Cnvme%5B0-9%5D%2Bn%5B0-9%5D%2B

BetterUptime:

- https://betteruptime.com/team/14149/monitors/389098

---

- docker

Also install docker (see https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository and https://docs.docker.com/engine/install/linux-postinstall/).

Launch a docker container with mongo:

```bash
docker run -p 27018:27017 --name datasets-preview-backend-mongo -d --restart always mongo:latest
```

Note that we assume `ASSETS_DIRECTORY=/data` in the nginx configuration. If you set the assets directory to another place, or let the default, ensure the nginx configuration is setup accordingly. Beware: the default directory inside `/home/hf/.cache` is surely not readable by the nginx user.

---

how to monitor the job runners and the queue?

---

```python
DEFAULT_MAX_SIZE_FALLBACK
DEFAULT_ROWS_MAX_BYTES,
DEFAULT_ROWS_MAX_NUMBER,
DEFAULT_ROWS_MIN_NUMBER,

# for tests - to be removed
MAX_SIZE_FALLBACK = get_int_value(os.environ, "MAX_SIZE_FALLBACK", DEFAULT_MAX_SIZE_FALLBACK)
ROWS_MAX_BYTES = get_int_value(d=os.environ, key="ROWS_MAX_BYTES", default=DEFAULT_ROWS_MAX_BYTES)
ROWS_MAX_NUMBER = get_int_value(d=os.environ, key="ROWS_MAX_NUMBER", default=DEFAULT_ROWS_MAX_NUMBER)
ROWS_MIN_NUMBER = get_int_value(d=os.environ, key="ROWS_MIN_NUMBER", default=DEFAULT_ROWS_MIN_NUMBER)
```

---

Warm the cache with:

```bash
pm2 start --no-autorestart --name warm make -- -C /home/hf/datasets-preview-backend/ warm
```
