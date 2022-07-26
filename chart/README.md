# datasets-server Helm chart

The `datasets-server` Helm [chart](https://helm.sh/docs/topics/charts/) describes the Kubernetes resources of the datasets-server application.

See the [helm.md](../docs_to_notion/helm.md) for some documentation about Helm and the Charts.

The cloud infrastructure for the datasets-server uses:

- Amazon ECR to store the docker images of the datasets-server services. See [docs/docker.md](../docs_to_notion/docker.md).
- Amazon EKS for the Kubernetes clusters. See [docs/kubernetes.md](../docs_to_notion/kubernetes.md).

Note that this Helm chart is used to manage the deployment of the `datasets-server` services to the cloud infrastructure (AWS) using Kubernetes. The infrastructure in itself is not created here, but in https://github.com/huggingface/infra/ using terraform. If you need to create or modify some resources, contact the infra team.

You might also be interested in reading the doc for [moon-landing](https://github.com/huggingface/moon-landing/blob/main/infra/hub/README.md).

## Deploy

To deploy to the `hub-ephemeral` Kubernetes cluster, ensure to first:

- install the [tools](../docs_to_notion/tools.md)
- [authenticate with AWS](../docs_to_notion/authentication.md)
- [select the `hub-ephemeral` cluster](../docs_to_notion/kubernetes.md#cluster)

Set the SHA of the last commit in [values.yaml](./values.yaml). It allows to select the adequate docker images in the ECR repositories (see the last build images at https://github.com/huggingface/datasets-server/actions/workflows/docker.yml).

Dry run:

```shell
make init
make diff-dev
```

Deploy:

```shell
make upgrade-dev
```
