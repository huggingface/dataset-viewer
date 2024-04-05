# Dataset viewer Helm chart

The dataset viewer Helm [chart](https://helm.sh/docs/topics/charts/) describes the Kubernetes resources of the dataset viewer application.

If you have access to the internal HF notion, see https://www.notion.so/huggingface2/Infrastructure-b4fd07f015e04a84a41ec6472c8a0ff5.

The cloud infrastructure for the dataset viewer uses:

- Docker Hub to store the docker images of the dataset viewer services.
- Amazon EKS for the Kubernetes clusters.

Note that this Helm chart is used to manage the deployment of the dataset viewer services to the cloud infrastructure (AWS) using Kubernetes. The infrastructure in itself is not created here, but in https://github.com/huggingface/infra/ using terraform. If you need to create or modify some resources, contact the infra team.

## Deploy

To deploy, go to https://cd.internal.huggingface.tech/applications.
