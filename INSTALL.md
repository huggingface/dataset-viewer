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
git fetch
git merge
# or better
# git checkout 0.2.0 # <- the latest release tag (https://github.com/huggingface/datasets-preview-backend/releases/latest)
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
sudo apt install python-is-python3 make nginx
```

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

[Install certbot](https://certbot.eff.org/lets-encrypt/ubuntufocal-nginx) with snap to manage the certificate for the domain name. Email: infra+letsencrypt@huggingface.co.

```bash
sudo certbot --nginx
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

Check if the app is accessible at https://datasets-preview.huggingface.tech/healthcheck.

Finally, ensure that pm2 will restart on reboot (see https://pm2.keymetrics.io/docs/usage/startup/):

```bash
pm2 startup
# and follow the instructions
```
