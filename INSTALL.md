# Install guide

## Docker

Install docker (see https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository and https://docs.docker.com/engine/install/linux-postinstall/)

```
make install
make start-from-local-code
```

To use the docker images already compiled using the CI:

```
make start-from-remote-images
```

## Without docker

We assume a machine with Ubuntu.

We need to prepare space on the disk for the assets, for example at `/data/assets`:

```
sudo mkdir -p /data/assets
sudo chown -R hf:www-data /data
sudo chmod -R 755 /data
```

We also need to have a mongo server:

- install docker (see https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository and https://docs.docker.com/engine/install/linux-postinstall/)
- launch a docker container with mongo:

  ```bash
  docker run -p 27017:27017 --name datasets-server-mongo -d --restart always mongo:latest
  ```

Install and deploy the API server with [services/api/INSTALL.md](./services/api/INSTALL.md), the admin server with [services/admin/INSTALL.md](./services/admin/INSTALL.md) and the workers with [services/worker/INSTALL.md](./services/worker/INSTALL.md).

## Upgrade

See the instructions in [services/api/INSTALL.md](./services/api/INSTALL.md#upgrade) and [services/worker/INSTALL.md](./services/worker/INSTALL.md#upgrade). Also migrate the databases if needed (see the [libcache migrations README](./libs/libcache/migrations/README.md)).

## Production

datasets-server is installed on a [kubernetes cluster](https://us-east-1.console.aws.amazon.com/eks/home?region=us-east-1#/clusters)

Grafana:

- https://grafana.huggingface.tech/dashboards/f/j1kRCJEnk/hub?query=Datasets%20server
- https://grafana.huggingface.tech/d/a164a7f0339f99e89cea5cb47e9be617/kubernetes-compute-resources-workload?orgId=1&refresh=10s&var-datasource=Prometheus%20EKS%20Hub%20Prod&var-cluster=&var-namespace=datasets-server&var-type=deployment&var-workload=datasets-server-prod-worker-splits

BetterUptime:

- https://betteruptime.com/team/14149/monitors/389098
- https://betteruptime.com/team/14149/monitors/691070
