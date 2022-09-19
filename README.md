# Datasets server

> Integrate into your apps over 10,000 datasets via simple HTTP requests, with pre-processed responses and scalability built-in.

Documentation: https://huggingface.co/docs/datasets-server

## Install and development setup

To develop or deploy, see [CONTRIBUTING.md](./CONTRIBUTING.md)

## Architecture

The application is distributed in several components.

[api](./services/api) is a web server that exposes the [API endpoints](https://huggingface.co/docs/datasets-server). Apart from some endpoints (`valid`, `is-valid`), all the responses are served from pre-computed responses. That's the main point of this project: generating these responses takes time, and the API server provides this service to the users.

The precomputed responses are stored in a Mongo database called "cache" (see [libcache](./libs/libcache)). They are computed by workers ([worker](./services/worker)) which take their jobs from a job queue stored in a Mongo database called "queue" (see [libqueue](./libs/libqueue)), and store the results (error or valid response) into the "cache".

The API service exposes the `/webhook` endpoint which is called by the Hub on every creation, update or deletion of a dataset on the Hub. On deletion, the cached responses are deleted. On creation or update, a new job is appended in the "queue" database.

Note that two job queues exist:

- `splits`: the job is to refresh a dataset, namely to get the list of [config](https://huggingface.co/docs/datasets/v2.1.0/en/load_hub#select-a-configuration) and [split](https://huggingface.co/docs/datasets/v2.1.0/en/load_hub#select-a-split) names, then to create a new job for every split
- `first-rows`: the job is to get the columns and the first 100 rows of the split

Note also that the workers create local files when the dataset contains images or audios. A shared directory (`ASSETS_DIRECTORY`) must therefore be provisioned with sufficient space for the generated files. The `/first-rows` endpoint responses contain URLs to these files, served by the API under the `/assets/` endpoint.

Hence, the working application has:

- one instance of the API service which exposes a port
- M instances of the `splits` worker and N instances of the `first-rows` worker (N should generally be higher than M)
- a Mongo server with two databases: "cache" and "queue"
- a shared directory for the assets

The application also has:

- a reverse proxy in front of the API to serve static files and proxy the rest to the API server
- an admin server to serve technical endpoints

## Environments

The following environments contain all the modules: reverse proxy, API server, admin API server, workers, and the Mongo database.

| Environment              | URL                                                  | Type              | How to deploy                                                        |
| ------------------------ | ---------------------------------------------------- | ----------------- | -------------------------------------------------------------------- |
| Production               | https://datasets-server.huggingface.co               | Helm / Kubernetes | `make upgrade-prod` in [chart](./chart)                              |
| Development              | https://datasets-server.us.dev.moon.huggingface.tech | Helm / Kubernetes | `make upgrade-dev` in [chart](./chart)                               |
| Local from remote images | http://localhost:8100                                | Docker compose    | `make start-from-remote-images` (fetches docker images from AWS ECR) |
| Local build              | http://localhost:8000                                | Docker compose    | `make start-from-local-code` (builds docker images)                  |

The Hugging Face Hub instance can be configured thanks to `HF_ENDPOINT`, so that the datasets server can access the Hub, a private Hub, or the instance dedicated to CI (https://hub-ci.huggingface.co/). The `HF_TOKEN` environment variable used by the workers to access the gated datasets must be set accordingly.

| Where             | `HF_ENDPOINT` (api, worker)                                             | `HF_TOKEN` (worker)                                          |
| ----------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| production        | https://huggingface.co/                                                 | Kubernetes secret                                            |
| development       | https://huggingface.co/                                                 | Kubernetes secret                                            |
| local docker      | https://huggingface.co/. Override with `HF_ENDPOINT=... make start-...` | Enable the gated datasets with `HF_TOKEN=... make start-...` |
| e2e               | https://hub-ci.huggingface.co/                                          | Hard-coded: `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`          |
| worker unit tests | https://hub-ci.huggingface.co/                                          | Hard-coded: `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`          |
