TODO: add the scripts

To warm the cache, ie. add all the missing Hugging Face datasets to the queue:

```bash
make warm
```

Warm the cache with:

```bash
pm2 start --no-autorestart --name warm make -- -C /home/hf/datasets-server/ warm
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

how to monitor the workers and the queue?

grafana doesn't have any data (see links in INSTALL.md)
