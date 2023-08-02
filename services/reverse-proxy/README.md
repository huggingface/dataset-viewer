# Datasets server - reverse proxy

> Reverse-proxy in front of the API

See [docker-compose-datasets-server.yml](../../tools/docker-compose-datasets-server.yml) for usage.

Note that the template configuration is located in [chart/nginx-templates/](../../chart/nginx-templates/) in order to be reachable by the Helm chart to deploy on Kubernetes.

The reverse proxy uses nginx:

- it serves the static assets directly (the API also serves them if required, but it's unnecessary to go through starlette for this, and it generates errors in Safari, see [1](https://github.com/encode/starlette/issues/950) and [2](https://developer.apple.com/library/archive/documentation/AppleApplications/Reference/SafariWebContent/CreatingVideoforSafarioniPhone/CreatingVideoforSafarioniPhone.html#//apple_ref/doc/uid/TP40006514-SW6))
- it serves the OpenAPI specification
- it proxies the other requests to the API

It takes various environment variables, all of them are mandatory:

- `ASSETS_DIRECTORY`: the directory that contains the static assets, eg `/assets`
- `HOST`: domain of the reverse proxy, eg `localhost`
- `PORT`: port of the reverse proxy, eg `80`
- `URL_ADMIN`= URL of the admin, eg `http://admin:8081`
- `URL_API`= URL of the API, eg `http://api:8080`
- `URL_ROWS`= URL of the rows service, eg `http://rows:8082`
- `URL_SEARCH`= URL of the search service, eg `http://search:8083`

The image requires three directories to be mounted (from volumes):

- `$ASSETS_DIRECTORY` (read-only): the directory that contains the static assets.
- `/etc/nginx/templates` (read-only): the directory that contains the nginx configuration template ([templates](./templates/))
- `/staticfiles` (read-only): the directory that contains the static files (`openapi.json`).
