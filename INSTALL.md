# Install guide

## Docker

Install docker (see https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository and https://docs.docker.com/engine/install/linux-postinstall/)

```
docker-compose up --build -d --scale splits-worker=5
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
  docker run -p 27017:27017 --name datasets-preview-backend-mongo -d --restart always mongo:latest
  ```

Install and deploy the API server with [services/api_service/INSTALL.md](./services/api_service/INSTALL.md) and the workers with [services/job_runner/INSTALL.md](./services/job_runner/INSTALL.md).

## Upgrade

See the instructions in [services/api_service/INSTALL.md](./services/api_service/INSTALL.md#upgrade) and [services/job_runner/INSTALL.md](./services/job_runner/INSTALL.md#upgrade). Also migrate the databases if needed (see the [libcache migrations README](./libs/libcache/migrations/README.md)).

## Production

datasets-preview-backend is installed on a t2.2xlarge [EC2 virtual machine](https://us-east-1.console.aws.amazon.com/ec2/v2/home?region=us-east-1#InstanceDetails:instanceId=i-0b19b8deb4301ad4a) (under the "JULIEN CHAUMOND" account).

```bash
ssh hf@ec2-54-209-89-185.compute-1.amazonaws.com
# or using https://github.com/huggingface/conf/blob/master/ssh-config-hf-aws
ssh aws/datasets-preview-backend/1

/: 500 GB

public ipv4: 54.209.89.185
private ipv4: 172.30.4.71
domain name: datasets-preview.huggingface.tech
```

Grafana (TODO: no data at the moment):

- https://grafana.huggingface.co/d/gBtAotjMk/use-method?orgId=2&var-DS_PROMETHEUS=HF%20Prometheus&var-node=data-preview
- https://grafana.huggingface.co/d/rYdddlPWk/node-exporter-full?orgId=2&refresh=1m&var-DS_PROMETHEUS=HF%20Prometheus&var-job=node_exporter_metrics&var-node=data-preview&var-diskdevices=%5Ba-z%5D%2B%7Cnvme%5B0-9%5D%2Bn%5B0-9%5D%2B

BetterUptime:

- https://betteruptime.com/team/14149/monitors/389098
