# datasets-server Helm chart

The `datasets-server` Helm [chart](https://helm.sh/docs/topics/charts/) describes the Kubernetes resources of the datasets-server application.

See the [helm.md](../../docs/helm.md) for some documentation about Helm and the Charts.

## Deploy

To deploy to the `hub-ephemeral` Kubernetes cluster, ensure to first:

- install the [tools](../../docs/tools.md)
- [authenticate with AWS](../../docs/authentication.md)
- [select the `hub-ephemeral` cluster](../../docs/kubernetes.md#cluster)

Set the SHA of the last commit in [values.yaml](./values.yaml). It allows to select the adequate docker images in the ECR repositories (see the last build images at https://github.com/huggingface/datasets-server/actions/workflows/docker.yml).

Dry run:

```shell
helm dependency update .
make diff-ephemeral
```

Deploy:

```shell
make upgrade-ephemeral
```
