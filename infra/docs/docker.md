# Docker images repositories

## Amazon Elastic Container Registry (ECR)

We use a private registry of docker images on Amazon Elastic Container Registry (ECR): https://us-east-1.console.aws.amazon.com/ecr/repositories?region=us-east-1.

The docker images are pushed there using the CI ([docker.yml](../.github/workflows/docker.yml)).

Every image is tagged with the git commit used to build it (short form, ie: `sha-698411e`).

The docker repositories are:

- `707930574880.dkr.ecr.us-east-1.amazonaws.com/hub-datasets-server-api` for the API service. See https://us-east-1.console.aws.amazon.com/ecr/repositories/private/707930574880/hub-datasets-server-api.
- `707930574880.dkr.ecr.us-east-1.amazonaws.com/hub-datasets-server-worker` for the worker. See https://us-east-1.console.aws.amazon.com/ecr/repositories/private/707930574880/hub-datasets-server-worker.

To create, modify or delete ECR repositories, ask the infra team.

If you want to list, pull or push a docker image manually, you have to login before:

```
aws ecr get-login-password --profile=hub | docker login --username AWS --password-stdin 707930574880.dkr.ecr.us-east-1.amazonaws.com
```

You can also use `aws ecr` to get the list of images of a repository, for example:

```
aws ecr list-images --profile=hub --repository-name=hub-datasets-server-api
aws ecr describe-images --profile=hub --repository-name=hub-datasets-server-api
```

The documentation for the `aws ecr` CLI is here: https://docs.aws.amazon.com/cli/latest/reference/ecr/index.html.
