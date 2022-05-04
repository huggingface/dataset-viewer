# Infra

## Description

The cloud infrastructure for the datasets-server uses:

- Amazon ECR to store the docker images of the datasets-server services. See [docs/docker.md](./docs/docker.md).
- Amazon EKS for the Kubernetes clusters. See [docs/kubernetes.md](./docs/kubernetes.md).

Before starting, ensure to:

- [install the tools](./docs/tools.md)
- [setup the AWS CLI profile](./docs/authentication.md)

Note that this directory (`infra/`) is used to manage the deployment of the `datasets-server` services to the cloud infrastructure (AWS) using Kubernetes. The infrastructure in itself is not created here, but in https://github.com/huggingface/infra/ using terraform. If you need to create or modify some resources, contact the infra team.

The subdirectories are:

- [docs/](./docs/): documentation
- [kube](./kube): the kubernetes configuration
