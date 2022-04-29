# INSTALL

## Requirements

The requirements are:

- node (for pm2)
- Python 3.9.6+
- Poetry 1.1.7+
- make
- nginx

We assume a machine running Ubuntu. Install packages:

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

## Install and configure

Install the API service:

```bash
cd
# See https://github.blog/2013-09-03-two-factor-authentication/#how-does-it-work-for-command-line-git for authentication
git clone https://github.com/huggingface/datasets-preview-backend.git
cd datasets-preview-backend/services/api_service
make install
```

Copy and edit the environment variables file:

```bash
cd datasets-preview-backend/services/api_service
cp .env.example .env
vi .env
```

Note that we assume `ASSETS_DIRECTORY=/data` in the nginx configuration. If you set the assets directory to another place, or let the default, ensure the nginx configuration is setup accordingly. Beware: the default directory inside `/home/hf/.cache` is surely not readable by the nginx user.

## Deploy

Launch the API with pm2:

```bash
pm2 start --name api make -- -C /home/hf/datasets-preview-backend/ run
```

Check if the api is accessible at https://datasets-preview.huggingface.tech/healthcheck.

Finally, ensure that pm2 will restart on reboot (see https://pm2.keymetrics.io/docs/usage/startup/):

- if it's the first time:
  ```bash
  pm2 startup
  # and follow the instructions
  ```
- else:
  ```bash
  pm2 save
  ```

## Manage

Use [pm2](https://pm2.keymetrics.io/docs/usage/quick-start/#cheatsheet) to manage the service.

```bash
pm2 list
pm2 logs api
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
cd services/api_service
pyenv install 3.9.6
pyenv local 3.9.6
poetry env use python3.9
```

Install packages

```
make install
```

Check if new environment variables are available and edit the environment variables in `.env`:

```
cd services/api_service
diff .env.example .env
vi .env
```

Apply the database migrations (see [libs/libcache/src/libcache/migrations/README.md](./../../libs/libcache/migrations/README.md)) if any (in this case: ensure to upgrade the other services too).

```
# see https://github.com/huggingface/datasets-preview-backend/blob/main/libs/libcache/migrations/README.md
```

If you want to be extra-sure, check that all the tests are passing

```
make test
```

Restart

```
pm2 restart api
```

Check if the API is accessible at https://datasets-preview.huggingface.tech/healthcheck.

Finally un-pause the monitor at https://betteruptime.com/team/14149/monitors/389098.
