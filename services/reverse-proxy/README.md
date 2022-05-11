# Datasets server - reverse proxy

> Reverse-proxy in front of the API

See [docker-compose.yml](../../docker-compose.yml) for usage.

Note that the template configuration is located in [infra/charts/datasets-server/nginx-templates/](../../infra/charts/datasets-server/nginx-templates/) in order to be reachable by the Helm chart to deploy on Kubernetes.

The reverse proxy uses nginx:

- it serves the static assets directly (the API also serves them if required, but it's unnecessary to go through starlette for this, and it generates errors in Safari, see [1](https://github.com/encode/starlette/issues/950) and [2](https://developer.apple.com/library/archive/documentation/AppleApplications/Reference/SafariWebContent/CreatingVideoforSafarioniPhone/CreatingVideoforSafarioniPhone.html#//apple_ref/doc/uid/TP40006514-SW6)
- it proxies the other requests to the API
- it caches all the API responses, depending on their `cache-control` header
- it sets the `Access-Control-Allow-Origin` header to `*` to allow cross-origin requests

It takes various environment variables, all of them are mandatory:

- `ASSETS_DIRECTORY`: the directory that contains the static assets, eg `/assets`
- `CACHE_INACTIVE`: maximum duration before being removed from cache, eg `24h` (see [proxy_cache_path](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_cache_path))
- `CACHE_MAX_SIZE`: maximum size of the cache, eg `1g` (see [proxy_cache_path](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_cache_path))
- `CACHE_DIRECTORY`: the directory that contains the nginx cache, eg `/nginx-cache`
- `CACHE_ZONE_SIZE`: size of the cache index, eg `50m` (see [proxy_cache_path](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_cache_path))
- `HOST`: domain of the reverse proxy, eg `localhost`
- `PORT`: port of the reverse proxy, eg `80`
- `TARGET_URL`= URL of the API, eg `http://api:8080`

The image requires three directories to be mounted (from volumes):

- `$ASSETS_DIRECTORY` (read-only): the directory that contains the static assets.
- `$CACHE_DIRECTORY` (read/write): the directory that contains the nginx cache
- `/etc/nginx/templates` (read-only): the directory that contains the nginx configuration template ([templates](./templates/))
