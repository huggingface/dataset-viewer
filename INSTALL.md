# INSTALL

datasets-preview-backend is installed on a virtual machine (ec2-54-158-211-3.compute-1.amazonaws.com).

## Manage

Use [pm2](https://pm2.keymetrics.io/docs/usage/quick-start/#cheatsheet) to manage the service.

```bash
pm2 list
pm2 logs
```

## Upgrade

To deploy a new version of datasets-preview-backend, first update the code

```
cd /home/hf/datasets-preview-backend/
git fetch
git merge
```

Install packages

```
make install
```

Restart

```
pm2 restart all
```

Check if the app is accessible at http://54.158.211.3/healthcheck.

## Machine

```bash
ssh hf@ec2-54-158-211-3.compute-1.amazonaws.com

/: 200 GB

ipv4: 172.30.4.71
ipv4 (public): 54.158.211.3
```

grafana:

- https://grafana.huggingface.co/d/gBtAotjMk/use-method?orgId=2&var-DS_PROMETHEUS=HF%20Prometheus&var-node=data-preview
- https://grafana.huggingface.co/d/rYdddlPWk/node-exporter-full?orgId=2&refresh=1m&var-DS_PROMETHEUS=HF%20Prometheus&var-job=node_exporter_metrics&var-node=data-preview&var-diskdevices=%5Ba-z%5D%2B%7Cnvme%5B0-9%5D%2Bn%5B0-9%5D%2B

## Install

Install packages, logged as `hf`:

```bash
sudo apt install python-is-python3 make nginx
```

Also install node and npm (with [nvm](https://github.com/nvm-sh/nvm)), then:

```bash
npm i -g pm2@latest
```

Also [install poetry](https://python-poetry.org/docs/master/#installation). Don't forget to add `poetry` to the `PATH` environment variable.

Configure nginx as a reverse-proxy to expose the application on the port 80:

/: 10 GB
/data: 100 GB

ipv4: 172.30.0.73
ipv6: fe80::1060:d6ff:feee:6d31
ipv4 (public): 3.83.96.81

````

grafana:

- https://grafana.huggingface.co/d/rYdddlPWk/node-exporter-full?orgId=2&refresh=1m&from=now-15m&to=now&var-DS_PROMETHEUS=HF%20Prometheus&var-job=node_exporter_metrics&var-node=tensorboard-launcher
- https://grafana.huggingface.co/d/gBtAotjMk/use-method?orgId=2&var-DS_PROMETHEUS=HF%20Prometheus&var-node=tensorboard-launcher

## Install tensorboard-launcher on the machine

Install packages, logged as `hf`:

```bash
sudo apt install git-lfs nginx python-is-python3
````

Also [install poetry](https://python-poetry.org/docs/master/#installation).

Install tensorboard-launcher:

```bash
cd
# See https://github.blog/2013-09-03-two-factor-authentication/#how-does-it-work-for-command-line-git for authentication
git clone https://github.com/huggingface/tensorboard-launcher.git
cd tensorboard-launcher
poetry install
```

Create a launcher script

```bash
vi ~/launch.sh
```

```bash
#!/bin/bash
cd /home/hf/tensorboard-launcher/
TBL_NGINX_BASE_PATH=/data/nginx/tensorboard TBL_MODELS_BASE_PATH=/data/models make run
```

```bash
chmod +x ~/launch.sh
```

Create a systemd service

```bash
sudo vi /etc/systemd/system/tensorboard.service
```

```
[Unit]
Description=Tensorboard Daemon
After=network-online.target

[Service]
Type=simple

User=hf
Group=hf
UMask=007

ExecStart=/home/hf/launch.sh
ExecStop=/bin/kill -9 $MAINPID

Restart=on-failure

# Configures the time to wait before service is stopped forcefully.
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable tensorboard
```

Configure nginx

```bash
sudo unlink /etc/nginx/sites-enabled/default
sudo vi /etc/nginx/sites-available/reverse-proxy.conf
```

```bash
server {
  listen 80;
  listen [::]:80;

  access_log /var/log/nginx/reverse-access.log;
  error_log /var/log/nginx/reverse-error.log;

  location / {
    proxy_pass http://localhost:8000/;
    proxy_set_header Host $proxy_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_http_version 1.1;
  }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/reverse-proxy.conf /etc/nginx/sites-enabled/reverse-proxy.conf
sudo nginx -t # Test
sudo systemctl reload nginx
```

Install datasets-preview-backend:

```bash
cd
# See https://github.blog/2013-09-03-two-factor-authentication/#how-does-it-work-for-command-line-git for authentication
git clone https://github.com/huggingface/datasets-preview-backend.git
cd datasets-preview-backend
make install
```

Launch the app with pm2:

```bash
PORT=8000 pm2 start --name datasets-preview-backend make -- -C /home/hf/datasets-preview-backend/ run
```

Check if the app is accessible at http://54.158.211.3/healthcheck.

Finally, ensure that pm2 will restart on reboot (see https://pm2.keymetrics.io/docs/usage/startup/):

```bash
pm2 startup
# and follow the instructions
```
