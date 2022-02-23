# INSTALL

datasets-preview-backend is installed on a virtual machine (ec2-54-158-211-3.compute-1.amazonaws.com).

## Manage

Use [pm2](https://pm2.keymetrics.io/docs/usage/quick-start/#cheatsheet) to manage the service.

```bash
pm2 list
pm2 logs
```

## Upgrade

To deploy a new version of datasets-preview-backend, first pause the monitor at https://betteruptime.com/team/14149/monitors/389098.

Then update the code

```
cd /home/hf/datasets-preview-backend/
git fetch --tags
git checkout XXXX # <- the latest release tag (https://github.com/huggingface/datasets-preview-backend/releases/latest)
```

If the Python version has been increased to 3.9.6, for example, [run](https://stackoverflow.com/a/65589331/7351594):

```
pyenv install 3.9.6
pyenv local 3.9.6
poetry env use python3.9
```

Install packages

```
make install
```

Check is new environment variables are available and edit the environment variables in `.env`:

```
diff .env.example .env
vi .env
```

Check that all the tests are passing

```
make test
```

Restart

```
pm2 restart all
```

Check if the app is accessible at https://datasets-preview.huggingface.tech/healthcheck.

Finally un-pause the monitor at https://betteruptime.com/team/14149/monitors/389098.

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

## Install

Install packages, logged as `hf`:

```bash
sudo apt install python-is-python3 make nginx libicu-dev ffmpeg libavcodec-extra libsndfile1
```

Also install docker (see https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository and https://docs.docker.com/engine/install/linux-postinstall/).

Also install node and npm (with [nvm](https://github.com/nvm-sh/nvm)), then:

```bash
npm i -g pm2@latest
```

Also [install poetry](https://python-poetry.org/docs/master/#installation). Don't forget to add `poetry` to the `PATH` environment variable.

Configure nginx as a reverse-proxy to expose the application on the port 80:

```bash
sudo unlink /etc/nginx/sites-enabled/default
sudo vi /etc/nginx/sites-available/reverse-proxy.conf
```

```bash
server {
  listen 80;
  listen [::]:80;
  server_name datasets-preview.huggingface.tech;

  add_header 'Access-Control-Allow-Origin' '*' always;

  access_log /var/log/nginx/reverse-access.log;
  error_log /var/log/nginx/reverse-error.log;

  # due to https://github.com/encode/starlette/issues/950, which generates errors in Safari: https://developer.apple.com/library/archive/documentation/AppleApplications/Reference/SafariWebContent/CreatingVideoforSafarioniPhone/CreatingVideoforSafarioniPhone.html#//apple_ref/doc/uid/TP40006514-SW6
  # we serve the static files from nginx instead of starlette
  location /assets/ {
    alias /data/assets/;
  }

  proxy_cache_path /data/nginx/cache levels=1:2 keys_zone=STATIC:50m inactive=24h max_size=1g;

  location / {
    proxy_pass http://localhost:8000/;
    proxy_set_header Host $proxy_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
    # cache all the HEAD+GET requests (without Set-Cookie)
    # Cache-Control is used to determine the cache duration
    # see https://www.nginx.com/blog/nginx-caching-guide/
    proxy_buffering on;
    proxy_cache STATIC;
    proxy_cache_use_stale error timeout invalid_header updating http_500 http_502 http_503 http_504;
    proxy_cache_background_update on;
    proxy_cache_lock on;
  }
}
```

```bash
sudo mkdir -p /data/assets /data/nginx/cache
sudo chmod -R a+x /data/assets /data/nginx/cache
sudo ln -s /etc/nginx/sites-available/reverse-proxy.conf /etc/nginx/sites-enabled/reverse-proxy.conf
sudo nginx -t # Test
sudo systemctl reload nginx
```

[Install certbot](https://certbot.eff.org/lets-encrypt/ubuntufocal-nginx) with snap to manage the certificate for the domain name. Email: infra+letsencrypt@huggingface.co.

```bash
sudo certbot --nginx
```

Launch a docker container with mongo:

```bash
docker run -p 27018:27017 --name datasets-preview-backend-mongo -d --restart always mongo:latest
```

Install datasets-preview-backend:

```bash
cd
# See https://github.blog/2013-09-03-two-factor-authentication/#how-does-it-work-for-command-line-git for authentication
git clone https://github.com/huggingface/datasets-preview-backend.git
cd datasets-preview-backend
make install
```

Copy and edit the environment variables file:

```bash
cp .env.example .env
vi .env
```

Note that we assume `ASSETS_DIRECTORY=/data` in the nginx configuration. If you set the assets directory to another place, or let the default, ensure the nginx configuration is setup accordingly. Beware: the default directory inside `/home/hf/.cache` is surely not readable by the nginx user.

Launch the app with pm2:

```bash
pm2 start --name app make -- -C /home/hf/datasets-preview-backend/ run
```

Check if the app is accessible at https://datasets-preview.huggingface.tech/healthcheck.

Warm the cache with:

```bash
pm2 start --no-autorestart --name warm make -- -C /home/hf/datasets-preview-backend/ warm
```

Setup workers (run again to create another worker, and so on):

```bash
WORKER_QUEUE=datasets pm2 start --name worker-datasets make -- -C /home/hf/datasets-preview-backend/ worker
WORKER_QUEUE=splits pm2 start --name worker-splits make -- -C /home/hf/datasets-preview-backend/ worker
```

Finally, ensure that pm2 will restart on reboot (see https://pm2.keymetrics.io/docs/usage/startup/):

```bash
pm2 startup
# and follow the instructions
```
