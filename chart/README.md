# datasets-server Helm chart

The `datasets-server` Helm [chart](https://helm.sh/docs/topics/charts/) describes the Kubernetes resources of the datasets-server application.

If you have access to the internal HF notion, see https://www.notion.so/huggingface2/Infrastructure-b4fd07f015e04a84a41ec6472c8a0ff5.

The cloud infrastructure for the datasets-server uses:

- Amazon ECR to store the docker images of the datasets-server services.
- Amazon EKS for the Kubernetes clusters.

Note that this Helm chart is used to manage the deployment of the `datasets-server` services to the cloud infrastructure (AWS) using Kubernetes. The infrastructure in itself is not created here, but in https://github.com/huggingface/infra/ using terraform. If you need to create or modify some resources, contact the infra team.

## Deploy

To deploy to the `hub-ephemeral` Kubernetes cluster, ensure to first:

- install the tools (aws, kubectl, helm)
- authenticate with AWS
- select the `hub-ephemeral` cluster

Dry run:

```shell
make init
make diff-dev
```

Deploy:

```shell
make upgrade-dev
```
