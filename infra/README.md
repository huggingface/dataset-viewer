# Infra

## Amazon Elastic Container Registry (ECR)

The docker images are pushed using the CI ([docker.yml](../.github/workflows/docker.yml)) to a private registry of docker images: https://us-east-1.console.aws.amazon.com/ecr/repositories?region=us-east-1.

Every image is tagged with the git commit used to build it (short form, ie: `sha-698411e`).

The docker repositories are:

- `707930574880.dkr.ecr.us-east-1.amazonaws.com/hub-datasets-server-api` for the API service. See https://us-east-1.console.aws.amazon.com/ecr/repositories/private/707930574880/hub-datasets-server-api.
- `707930574880.dkr.ecr.us-east-1.amazonaws.com/hub-datasets-server-worker` for the worker. See https://us-east-1.console.aws.amazon.com/ecr/repositories/private/707930574880/hub-datasets-server-worker.

To create, modify or delete ECR repositories, ask the infra team.
